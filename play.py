# standard lib
import os
import math
from glob import glob
import gymnasium as gym
from gymnasium.envs.registration import register
from gym.wrappers import RecordVideo
from IPython.display import Video, display, clear_output
from tqdm import tqdm
import argparse
import torch 

# this library
from rnd_rl.runner.policy_runner import PPOConfig, PolicyRunner
from rnd_rl.utils.sim_util import visualize
from rnd_rl.utils.util import set_seed

# torch default device
# NOTE: qpth does not support mps backend, so disable it. 
# if  torch.backends.mps.is_available():
#     device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
torch.set_default_device(device)
print(f"Using device: {device}")
set_seed(42)

"""
register gym envs
"""

register(
    id="CustomInvertedPendulum-v0",
    entry_point="rnd_rl.env.env:CustomInvertedPendulum",
    )


def play(
    env_name:str="InvertedPendulum-v5",
    num_envs:int=64,
    max_epochs:int=250,
    experiment_name:str="PPO",
    use_rnd:bool=True,
    reward_normalization:bool=True,
    obs_normalization:bool=True,
    enable_safety_layer:bool=True,
    ):
    
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_name, reset_noise_scale=0.2) for _ in range(num_envs)]
        )
    
    policy_cfg = PPOConfig(
        use_rnd=use_rnd, 
        clip_params=0.2,
        init_noise_std=1.0, 
        reward_normalization=reward_normalization,
        obs_normalization=obs_normalization,
        enable_safety_layer=enable_safety_layer
    )
    
    policy_runner = PolicyRunner(
        envs=envs, 
        policy_cfg=policy_cfg, 
        num_mini_epochs=10,
        device=device, 
        experiment_name=experiment_name, 
        enable_logging=False
    )
    
    # grab latest model 
    model_path = sorted(
        glob(f'runs/{policy_runner.alg.name}/model/model_*.pt'),
        key=lambda x: int(x.split('_')[-1].split('.pt')[0])
        )[-1]
    policy_runner.load(
        model_path, 
        map_location=device, 
        load_optimizer=False, 
        )
    
    visualize(policy_runner.alg, video_dir=f"./videos/{experiment_name}", device=device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="InvertedPendulum-v5", help="Name of the Gym environment")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--max_epochs", type=int, default=250, help="Maximum number of training epochs")
    parser.add_argument("--experiment_name", type=str, default="PPO", help="Name of the experiment")
    parser.add_argument("--use_rnd", action='store_true', help="Whether to use RND for intrinsic motivation")
    parser.add_argument("--normalize_rnd", action='store_true', help="Whether to normalize rewards")
    parser.add_argument("--enable_safety_layer", action='store_true', help="Whether to enable safety layer")
    
    args = parser.parse_args()
    
    play(
        env_name=args.env_name,
        num_envs=args.num_envs,
        max_epochs=args.max_epochs,
        experiment_name=args.experiment_name,
        use_rnd=args.use_rnd,
        reward_normalization=args.normalize_rnd,
        obs_normalization=args.normalize_rnd,
        enable_safety_layer=args.enable_safety_layer
    )