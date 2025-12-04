import os
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch 
import wandb
# from torch.utils.tensorboard import SummaryWriter

from rnd_rl.storage.trajectory_data import TrajData
from rnd_rl.algorithm.ppo import PPOAgent, PPOConfig
from rnd_rl.utils.wandb_util import WandbSummaryWriter


class PolicyRunner:
    def __init__(
        self, 
        envs: gym.vector.SyncVectorEnv,
        policy_cfg: PPOConfig,
        n_envs: int = 64,
        num_mini_epochs:int=10,
        num_steps_per_env: int = 256,
        device: torch.device = torch.device("cpu"),
        experiment_name: str = "PPO",
        save_interval: int = 10,
        enable_logging: bool = True,
        dict_obs_space = False # intended for some gymnasium_robotics envs. But here only maze.
        ):

        self.n_envs = n_envs # parallel envs 
        self.n_steps = num_steps_per_env # horizon length
        self.n_obs = envs.observation_space.shape[1] if not dict_obs_space \
            else envs.observation_space['observation'].shape[1]
        self.n_actions = envs.action_space.shape[1]
        self.num_mini_epochs = num_mini_epochs
        self.save_interval = save_interval
        self.dict_obs_space = dict_obs_space

        self.envs = envs
        self.traj_data = TrajData(self.n_steps, self.n_envs, self.n_obs, n_actions=self.n_actions) 
        self.policy_cfg = policy_cfg
        self.alg = PPOAgent(
            policy_cfg=policy_cfg,
            obs_dim=self.n_obs,
            action_dim=self.n_actions,
            device=device,
            experiment_name=experiment_name,
        )

        self.log_dir = f'runs/{self.alg.name}'
        if enable_logging:
            self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg={"wandb_project": "rnd_rl"})


        self.current_learning_epoch = 0


    def rollout(self, it:int):

        obs, _ = self.envs.reset()
        if self.dict_obs_space: obs = obs["observation"]
        obs = torch.Tensor(obs)

        for t in range(self.n_steps):
            # PPO doesnt use gradients here, but REINFORCE and VPG do.
            with torch.no_grad():
                actions, probs = self.alg.get_action(obs.to(self.alg.device))
            log_probs = probs.log_prob(actions).sum(dim=-1)
            next_obs, rewards, done, truncated, infos = self.envs.step(actions.to('cpu').numpy())
            if self.dict_obs_space: next_obs = next_obs["observation"]
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
        
        self.writer.add_scalar("Reward", self.traj_data.rewards.mean().item(), it)
        self.writer.add_scalar("Extrinsic Reward", self.traj_data.extrinsic_rewards.mean().item(), it)
        
        if self.dict_obs_space: return # specific for this project, when this is true we are using the maze env
        # plot cart position 
        fig, ax = plt.subplots(dpi=150)
        ax.plot(self.traj_data.states[:, 0, 0].cpu().numpy())
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Cart Position [m]")
        self.writer.add_figure("State/Cart_Position_0", fig, it)
        plt.close(fig)
        
        # plot cart velocity
        fig, ax = plt.subplots(dpi=150)
        ax.plot(self.traj_data.states[:, 0, 2].cpu().numpy())
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Cart Velocity [m/s]")
        self.writer.add_figure("State/Cart_Velocity_0", fig, it)
        plt.close(fig)
        
        # plot cart constraint violation
        cart_pos = self.traj_data.states[:, :, 0].cpu().numpy()
        cart_pos_limit = 0.7 
        cart_pos_violation = ((np.abs(cart_pos) - cart_pos_limit) > 0).mean() # mean over horizon and env
        self.writer.add_scalar("State/Cart_Position_Violation", cart_pos_violation, it)


    def update(self, it:int):
        self.current_learning_epoch = it
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
        
        # Save model
        if it % self.save_interval == 0:
            self.save(os.path.join(self.log_dir, "model", f"model_{it}.pt"))  # type: ignore

    def save(self, path: str) -> None:
        # Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_epoch,
        }
        # Save RND model if used
        if self.policy_cfg.use_rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging services
        self.writer.save_model(path, self.current_learning_epoch)
        
    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> None:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # Load RND model if used
        if self.policy_cfg.use_rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # Load optimizer if used
        if load_optimizer and resumed_training:
            # Algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # RND optimizer if used
            if self.policy_cfg.use_rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # Load current learning epoch
        if resumed_training:
            self.current_learning_epoch = loaded_dict["iter"]
