# import gymnasium as gym
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv
# import torch
import numpy as np
class CustomInvertedPendulum(InvertedPendulumEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # print("Inverted Pendulum environment with custom rewards initialized.")
        
    # def step(self, action):
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     cart_pos = obs[0]
    #     cart_pos_thresh = 0.7 
    #     cart_pos_penalty = np.maximum(0, np.abs(cart_pos) - cart_pos_thresh) * 2 
    #     reward = reward - cart_pos_penalty

    #     return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, state, action):
        reward = super()._compute_reward(state, action)
        
        cart_pos = state[0]
        cart_pos_thresh = 0.7 
        cart_pos_penalty = np.maximum(0, np.abs(cart_pos) - cart_pos_thresh) * 2 
        penalty = -cart_pos_penalty
        
        return reward + penalty