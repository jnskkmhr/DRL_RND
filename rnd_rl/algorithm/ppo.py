from dataclasses import dataclass, field
import torch 
import torch.nn as nn
from rnd_rl.storage.trajectory_data import TrajData
from rnd_rl.modules.actor_critic import ActorCritic
from rnd_rl.modules.rnd import RandomNetworkDistillation
from qpth.qp import QPFunction

@dataclass
class PPOConfig:
    actor_hidden_dims: list = field(default_factory=lambda: [128, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [128, 128])
    rnd_hidden_dims: list = field(default_factory=lambda: [128, 128])
    rnd_output_dim: int = 128
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    lr: float = 3e-4
    rnd_lr: float = 1e-4
    use_rnd: bool = True
    clip_params:float=0.1 
    gamma:float=0.99 
    gae_lambda:float=0.95
    obs_normalization: bool = False # for now RND and ActorCritic both normalize obs or not
    reward_normalization: bool = False
    safety_layer_enabled: bool = False
class PPOAgent(nn.Module):
    def __init__(
        self,
        policy_cfg: PPOConfig,
        obs_dim: int,
        action_dim: int,
        device: torch.device = torch.device("cpu"),
        safety_layer_enabled: bool = False,
        experiment_name: str = "PPO",
    ):
        super(PPOAgent, self).__init__()

        self.clip_params = policy_cfg.clip_params
        self.gamma = policy_cfg.gamma
        self.gae_lambda = policy_cfg.gae_lambda
        self.device = device
        
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dims=policy_cfg.actor_hidden_dims,
            critic_hidden_dims=policy_cfg.critic_hidden_dims,
            init_noise_std=policy_cfg.init_noise_std,
            noise_std_type=policy_cfg.noise_std_type,
            device=device,
            obs_normalization = policy_cfg.obs_normalization,
        ).to(self.device)
        self.safety_layer_enabled = safety_layer_enabled
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_cfg.lr)

        if policy_cfg.use_rnd:
            self.rnd = RandomNetworkDistillation(
                input_dim=obs_dim,
                output_dim=policy_cfg.rnd_output_dim,
                hidden_dims=policy_cfg.rnd_hidden_dims,
                device=device,
                obs_normalization = policy_cfg.obs_normalization,
                reward_normalization = policy_cfg.reward_normalization
            ).to(self.device)
            self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=policy_cfg.rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None
            
        if policy_cfg.use_rnd:
            self.name = "PPO_RND"
        else:
            self.name = "PPO"
        self.name = experiment_name
        
    def dpp_projection(self, a_raw, state, c_funcs, 
                   steps=10, eta=0.1, lambda_penalty=10.0):
        """
        a_raw: raw PPO action (requires_grad=True)
        c_funcs: list of constraint functions c_i(s, a)
        """

        x = a_raw.clone().requires_grad_(True)

        for _ in range(steps):
            penalty = 0.0

            for c_fn in c_funcs:
                c_val = c_fn(state, x)
                # soft penalty: smooth, differentiable
                penalty += torch.sum(torch.relu(c_val)**2)
            
            loss = (x - a_raw).pow(2).sum() + lambda_penalty * penalty

            # compute gradient
            grad = torch.autograd.grad(
                loss, x, create_graph=True
            )[0]

            # gradient descent update
            x = (x - eta * grad).detach().requires_grad_(True)
        return x

    def compute_cbf_constraints(self, states, h_functions, f_functions, g_functions, alpha=0.1):
        """
        states: (N, obs_dim) torch tensor
        h_functions: list of functions h_i(s)
        f_functions: function f(s)
        g_functions: function g(s)
        Returns:
            A: (N, m) torch tensor
            b: (N,) torch tensor
        """
        
        sta = states.clone().requires_grad_(True)

        f = f_functions(sta)
        g = g_functions(sta)

        A = []
        b = []
        for i, h_fn in enumerate(h_functions):
            h = h_fn(sta)
            dh_ds = torch.autograd.grad(
                h, sta, grad_outputs=torch.ones_like(h), create_graph=True
            )[0]

            h_f_dot = dh_ds @ f
            h_g_dot = dh_ds @ g

            A_i = h_g_dot
            b_i = -h_f_dot - alpha * h

            A.append(A_i)
            b.append(b_i)

        A = torch.stack(A, dim=0)
        b = torch.stack(b, dim=0)
        return A, b

    def cbf_qp_layer(self, a_raw, A, b, eps=1e-4):
        # a_raw: (m,) torch tensor
        # A: (k, m) torch tensor
        # b: (k,) torch tensor

        m = a_raw.shape[0]
        k = A.shape[0]

        Q = torch.eye(m, device=self.device)
        p = -a_raw

        G = -A
        h = -b

        A_eq = torch.zeros((0, m), device=self.device)
        b_eq = torch.zeros((0), device=self.device)

        a_safe = QPFunction(eps=eps)(Q, p, G, h, A_eq, b_eq)

        return a_safe
        
    def get_policy_loss(self, traj_data:TrajData)->torch.Tensor:
        predicted_values = self.policy.evaluate(traj_data.states).squeeze(-1)
        returns = traj_data.returns
        loss_fn = nn.MSELoss()
        value_loss = loss_fn(predicted_values, traj_data.returns.detach()).mean()
        _, probs = self.get_action(traj_data.states)
        log_probs = probs.log_prob(traj_data.actions).sum(dim=-1)
        old_log_probs = traj_data.log_probs.detach()
        ratio = torch.exp(log_probs - old_log_probs)
        advantage = returns - predicted_values
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_params, 1 + self.clip_params)
        policy_loss = -torch.min(ratio * advantage.detach(), clipped_ratio * advantage.detach()).mean()
        loss = value_loss + policy_loss
        return loss
    
    def get_rnd_loss(self, traj_data:TrajData)->torch.Tensor:
        predicted_embeddings = self.rnd.predictor(traj_data.states)
        target_embeddings = self.rnd.target(traj_data.states).detach()
        loss_fn = nn.MSELoss()
        rnd_loss = loss_fn(predicted_embeddings, target_embeddings)
        return rnd_loss
    
    def get_action(self, obs):
        # print("obs device ", obs.device,"self device ", self.device)
        actions = self.policy.act(obs)
        dist = self.policy.distribution
        if self.safety_layer_enabled:
            # Define constraint functions
            def c1(s, a):
                return s - 1.0  # s <= 1.0

            def c2(s, a):
                return -s - 1.0  # s >= -1.0

            c_funcs = [c1, c2]
            # Project action using DPP
            actions = self.dpp_projection(actions, obs, c_funcs)
            
        return actions, dist