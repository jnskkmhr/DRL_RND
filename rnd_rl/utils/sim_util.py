import os
import gymnasium as gym
from gym.wrappers import RecordVideo
from IPython.display import Video, display, clear_output
import torch

# @title Visualization code. Used later.

def visualize(agent, device=torch.device("cpu"), video_dir:str="./videos",
              env_name = "InvertedPendulum-v5"):

    os.makedirs(video_dir, exist_ok=True)

    using_maze_env = env_name == "PointMaze_Medium-v3" # special case.

    # Create environment with proper render_mode
    if using_maze_env:
        # force a longer trajectory where RND significantly outperforms vanilla PPO
        fixed_goal_maze =  [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, "g", 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, "r", 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]] 
        env = gym.make("PointMaze_Medium-v3", render_mode="rgb_array", 
                       maze_map = fixed_goal_maze, continuing_task = False)
    else:
        env = gym.make("InvertedPendulum-v5", render_mode="rgb_array", reset_noise_scale=0.2)
    # env = gym.make("InvertedPendulum-v5", render_mode="rgb_array", reset_noise_scale=0.2)

    # Apply video recording wrapper
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)

    obs, _ = env.reset()
    if using_maze_env: obs = obs["observation"]

    for t in range(4096):
        actions, _ = agent.get_action(torch.Tensor(obs)[None, :].to(device))
        actions = actions.detach()
        obs, _, done, _, _ = env.step(actions.squeeze(0).cpu().numpy())
        if using_maze_env: obs = obs["observation"]

        if done:
            break

    env.close()

    # Display the latest video
    video_path = os.path.join(video_dir, sorted(os.listdir(video_dir))[-1])  # Get the latest video


    clear_output(wait=True)
    display(Video(video_path, embed=True))