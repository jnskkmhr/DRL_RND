# Random Network Distillation 

Code under development. The code is based on provided assignments and https://github.com/leggedrobotics/rsl_rl

## Before start
If you are running on colab: 
* No installation is needed. You can either download the train_colab.ipynb file and open it in colab, or go to colab-> Open notebook -> GitHub and then paste the URL of this repository. Note that we cannot control the exact environment of colab and most of the testing are done on the local machine.

If you are running on your local machine: 
* Install uv (a package manager) if you do not have it.
* Head to requirements.txt and change the torch version according to your cuda installation, then run the installation commands below. Note that the virtual environment can occupy more than 5 GB of disk space.

## Installation 
```bash 
uv venv drl_env --python 3.11
source drl_env/bin/activate
uv pip install -r requirements.txt
```

## Training and inference
Go to train.ipynb if you are running on your local machine. 

Go to train_colab.ipynb if you are running on colab.