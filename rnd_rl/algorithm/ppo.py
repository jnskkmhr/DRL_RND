import torch 
import torch.nn as nn
from rnd_rl.storage.trajectory_data import TrajData
from rnd_rl.modules.actor_critic import ActorCritic
from rnd_rl.modules.rnd import RandomNetworkDistillation

class PPOAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: list = [128, 128],
        critic_hidden_dims: list = [128, 128],
        rnd_hidden_dims: list = [128, 128],
        rnd_output_dim: int = 128,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        lr: float = 3e-4,
        rnd_lr: float = 1e-4,
        clip_params:float=0.1, 
        gamma:float=0.99, 
        gae_lambda:float=0.95,
        device: torch.device = torch.device("cpu"),
    ):
        super(PPOAgent, self).__init__()
        
        self.clip_params = clip_params
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            device=device,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.rnd = RandomNetworkDistillation(
            input_dim=obs_dim,
            output_dim=rnd_output_dim,
            hidden_dims=rnd_hidden_dims,
            device=device,
        ).to(self.device)
        
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        
    def get_loss(self, traj_data:TrajData)->torch.Tensor:
        predicted_values = self.value(traj_data.states).squeeze(-1)
        returns = traj_data.returns
        loss_fn = nn.MSELoss()
        value_loss = loss_fn(predicted_values, traj_data.returns.detach()).mean()
        _, probs = self.get_action(traj_data.states)
        log_probs = probs.log_prob(traj_data.actions)
        old_log_probs = traj_data.log_probs.detach()
        ratio = torch.exp(log_probs - old_log_probs)
        advantage = returns - predicted_values
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_params, 1 + self.clip_params)
        policy_loss = -torch.min(ratio * advantage.detach(), clipped_ratio * advantage.detach()).mean()
        loss = value_loss + policy_loss
        return loss
    
    def get_action(self, obs):
        actions = self.policy.act(obs)
        dist = self.policy.distribution
        return actions, dist