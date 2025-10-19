import gymnasium as gym
import torch 
from torch.utils.tensorboard import SummaryWriter

from rnd_rl.storage.trajectory_data import TrajData
from rnd_rl.algorithm.ppo import PPOAgent


class PolicyRunner:
    def __init__(self):

        self.n_envs = 64
        self.n_steps = 256
        self.n_obs = 4

        self.envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(self.n_envs)])

        self.traj_data = TrajData(self.n_steps, self.n_envs, self.n_obs, n_actions=1) # 1 action choice is made
        self.agent = Agent(self.n_obs, n_actions=2)  # 2 action choices are available
        self.optimizer = Adam(self.agent.parameters(), lr=1e-3)
        self.writer = SummaryWriter(log_dir=f'runs/{self.agent.name}')


    def rollout(self, i):

        obs, _ = self.envs.reset()
        obs = torch.Tensor(obs)

        for t in range(self.n_steps):
            # PPO doesnt use gradients here, but REINFORCE and VPG do.
            with torch.no_grad() if self.agent.name == 'PPO' else torch.enable_grad():
                actions, probs = self.agent.get_action(obs)
            log_probs = probs.log_prob(actions)
            next_obs, rewards, done, truncated, infos = self.envs.step(actions.numpy())
            done = done | truncated  # episode doesnt truncate till t = 500, so never
            self.traj_data.store(t, obs, actions, rewards, log_probs, done)
            obs = torch.Tensor(next_obs)

        self.traj_data.calc_returns()

        self.writer.add_scalar("Reward", self.traj_data.rewards.mean(), i)
        self.writer.flush()


    def update(self):

        # A primary benefit of PPO is that it can train for
        # many epochs on 1 rollout without going unstable
        epochs = 10 if self.agent.name == 'PPO' else 1

        for _ in range(epochs):

            loss = self.agent.get_loss(self.traj_data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.traj_data.detach()
