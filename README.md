# Random Network Distillation 

This repository provides implementation of proximal policy optimization (PPO) and random network distillation (RND) for CS8803 Deep Reinforcement Learning course. It is based on [1] [(repository link)](https://github.com/leggedrobotics/rsl_rl) by the same authors of the original RND paper.

## Before start

Tested environment
* MacOS (Apple silicon)
* Ubuntu 22.04/24.04

### Run on your local machine
Install uv (a package manager) if you do not have it.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation 
Setup uv venv and install required python packages. 
```bash 
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Weight & biases setup
We use [weight & biases (wandb)](https://wandb.ai/site/) instead of tensorboard to log training data. \
Please make sure you have your wandb account and login in the local machine ([reference](https://docs.wandb.ai/models/quickstart#install-the-wandb-library-and-log-in))

```bash
source .venv/bin/activate
wandb login
```

## Training and inference
<!-- Go to `train.ipynb` if you are running on your local machine. \
Go to `train_colab.ipynb` if you are running on colab. -->
### Running in terminal (**recommended**)

```bash
# --- baseline PPO ---
# training
uv run train.py --experiment_name PPO

# inference
uv run play.py --experiment_name PPO

# --- w/ RND, normalization ---
# training
uv run train.py --experiment_name PPO_RND --use_rnd --normalize_rnd

# inference
uv run play.py --experiment_name PPO_RND --use_rnd --normalize_rnd

# --- w/ RND, normalization, safety shileding ---
# training
uv run train.py --experiment_name PPO_CBF --use_rnd --normalize_rnd --enable_safety_layer

# inference
uv run play.py --experiment_name PPO_CBF --use_rnd --normalize_rnd --enable_safety_layer

# --- baseline PPO in maze environment ---
# training
uv run train.py --env_name PointMaze_Medium-v3 --experiment_name maze_PPO

# inference
uv run play.py --env_name PointMaze_Medium-v3 --experiment_name maze_PPO

# --- w/ RND, normalization in maze environment ---
# training
uv run train.py --env_name PointMaze_Medium-v3 --experiment_name maze_PPO_RND --use_rnd --normalize_rnd

# inference
uv run play.py --env_name PointMaze_Medium-v3 --experiment_name maze_PPO_RND --use_rnd --normalize_rnd

```
### Running in notebook
Go to `demo.ipynb`. The notebook is an interactive walkthrough of training and visualization. **Certain devices may have the risk of jupyter kernel crashing when running training cells**. The notebook is equivalent to running the above commands and `plot.ipynb`.

## Plot results
Go to `plot.ipynb`


## Reference
```bibtex
@article{burda2018exploration,
  title={Exploration by random network distillation},
  author={Burda, Yuri and Edwards, Harrison and Storkey, Amos and Klimov, Oleg},
  journal={arXiv preprint arXiv:1810.12894},
  year={2018}
}

% RL + CBF
@article{emam2022safe,
  title={Safe reinforcement learning using robust control barrier functions},
  author={Emam, Yousef and Notomista, Gennaro and Glotfelter, Paul and Kira, Zsolt and Egerstedt, Magnus},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  publisher={IEEE}
}
```