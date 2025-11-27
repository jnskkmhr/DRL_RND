import gymnasium as gym
import torch 
from torch.utils.tensorboard import SummaryWriter

from rnd_rl.storage.trajectory_data import TrajData
from rnd_rl.algorithm.ppo import PPOAgent, PPOConfig


class PolicyRunner:
    def __init__(
        self, 
        envs: gym.vector.SyncVectorEnv,
        policy_cfg: PPOConfig,
        n_envs: int = 64,
        num_mini_epochs:int=10,
        num_steps_per_env: int = 256,
        device: torch.device = torch.device("cpu"),
        ):

        self.n_envs = n_envs # parallel envs 
        self.n_steps = num_steps_per_env # horizon length
        self.n_obs = envs.observation_space.shape[1]
        self.n_actions = envs.action_space.shape[1]
        self.num_mini_epochs = num_mini_epochs

        # self.envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(self.n_envs)])
        self.envs = envs
        self.traj_data = TrajData(self.n_steps, self.n_envs, self.n_obs, n_actions=self.n_actions) 
        self.policy_cfg = policy_cfg
        self.alg = PPOAgent(
            policy_cfg=policy_cfg,
            obs_dim=self.n_obs,
            action_dim=self.n_actions,
            device=device,
            safety_layer_enabled=policy_cfg.safety_layer_enabled,
        )

        self.writer = SummaryWriter(log_dir=f'runs/{self.alg.name}')


    def rollout(self, i):

        obs, _ = self.envs.reset()
        obs = torch.Tensor(obs)

        for t in range(self.n_steps):
            # PPO doesnt use gradients here, but REINFORCE and VPG do.
            if self.policy_cfg.safety_layer_enabled:
                actions, probs = self.alg.get_action(obs.to(self.alg.device))
            else:
                with torch.no_grad():
                    actions, probs = self.alg.get_action(obs.to(self.alg.device))
            log_probs = probs.log_prob(actions).sum(dim=-1)
            actions = actions.detach()
            next_obs, rewards, done, truncated, infos = self.envs.step(actions.to('cpu').numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            self.traj_data.store(t, obs, actions, rewards, log_probs, done)
            obs = torch.Tensor(next_obs)
            
            # update observation and rnd normalizers value here, and also normalize obs and reward
            # like in process_env_step in on_policy_runner.py
            self.alg.policy.update_normalization(obs.to(self.alg.device))
            # obs normalizer applied to actor in act, and to critic when evaluate (computing returns)
            if self.policy_cfg.use_rnd:
                self.alg.rnd.update_normalization(obs.to(self.alg.device))
                intrinsic_rewards = self.alg.rnd.get_intrinsic_reward(obs.to(self.alg.device)).to(self.alg.device)
                self.traj_data.rewards[t] += intrinsic_rewards

        last_value = self.alg.policy.evaluate(obs.to(self.alg.device)).detach()
        values = self.alg.policy.evaluate(self.traj_data.states).detach().squeeze()
        self.traj_data.calc_returns(values, last_value=last_value)

        self.writer.add_scalar("Reward", self.traj_data.rewards.mean(), i)
        self.writer.add_scalar("Extrinsic Reward", self.traj_data.extrinsic_rewards.mean(), i)
        self.writer.flush()


    def update(self):

        # A primary benefit of PPO is that it can train for
        # many epochs on 1 rollout without going unstable
        for _ in range(self.num_mini_epochs):

            policy_loss = self.alg.get_policy_loss(self.traj_data)
            if self.alg.rnd:
                rnd_loss = self.alg.get_rnd_loss(self.traj_data)

            # compute gradients
            self.alg.optimizer.zero_grad()
            policy_loss.backward()
            if self.alg.rnd:
                self.alg.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            
            # apply gradients
            self.alg.optimizer.step()
            if self.alg.rnd_optimizer:
                self.alg.rnd_optimizer.step()

        self.traj_data.detach()
