# Random Network Distillation 

<<<<<<< HEAD
Code under development. The code is based on provided assignments and https://github.com/leggedrobotics/rsl_rl

## Before start
If you are running on colab: 
* No installation is needed. You can either download the train_colab.ipynb file and open it in colab, or go to colab-> Open notebook -> GitHub and then paste the URL of this repository. Note that we cannot control the exact environment of colab and most of the testing are done on the local machine.

If you are running on your local machine: 
* Install uv (a package manager) if you do not have it.
* Head to requirements.txt and change the torch version according to your cuda installation, then run the installation commands below. Note that the virtual environment can occupy more than 5 GB of disk space.
=======
This repository provides implementation of proximal policy optimization (PPO) and random network distillation (RND) for CS8803 Deep Reinforcement Learning course. 

## Before start

Tested environment
* MacOS (Apple silicon)
* Ubuntu 22.04

### Run on your local machine (**recommended**)
Install uv (a package manager) if you do not have it.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

###  Running on colab
No installation is needed. 
You can either download the train_colab.ipynb file and open it in colab, or go to colab-> Open notebook -> GitHub and then paste the URL of this repository. 
Note that we cannot control the exact environment of colab and most of the testing are done on the local machine.
>>>>>>> main

## Installation 
Setup uv venv and install required python packages. 
```bash 
uv venv drl_env --python 3.11
source drl_env/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Weight & biases setup
We use [weight & biases (wandb)](https://wandb.ai/site/) instead of tensorboard to log training data. \
Please make sure you have your wandb account and login in the local machine ([reference](https://docs.wandb.ai/models/quickstart#install-the-wandb-library-and-log-in))

```
source drl_env/bin/activate
wandb login
```

## Training and inference
<<<<<<< HEAD
Go to train.ipynb if you are running on your local machine. 

Go to train_colab.ipynb if you are running on colab.
=======
Go to `train.ipynb` if you are running on your local machine. \
Go to `train_colab.ipynb` if you are running on colab.

## Plot results
Go to `plot.ipynb`
>>>>>>> main
