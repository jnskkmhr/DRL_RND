import torch
import torch.nn as nn
from rnd_rl.modules.normalizer import ObsNormalizer, RewardNormalizer

class RandomNetworkDistillation(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [128, 128],
        device: torch.device = torch.device("cpu"),
        obs_normalization: bool = False,
        reward_normalization: bool = False,
        reward_scale: float = 1.0 # overly high intrinsic rew. can hurt training
    ):
        super(RandomNetworkDistillation, self).__init__()
        
        self.device = device
        self.reward_scale = reward_scale
        
        # Build the target network (fixed)
        target_layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            target_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                target_layers.append(nn.ReLU())
        self.target = nn.Sequential(*target_layers).to(self.device)
        
        # Build the predictor network (trainable)
        predictor_layers = []
        for i in range(len(dims) - 1):
            predictor_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                predictor_layers.append(nn.ReLU())
        self.predictor = nn.Sequential(*predictor_layers).to(self.device)
        
        # make target network not trainable
        self.target.eval()

        # normalization, if specified
        self.obs_normalization = obs_normalization
        self.reward_normalization = reward_normalization
        if obs_normalization:
            self.obs_normalizer = ObsNormalizer(obs_dim=[input_dim], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity() # to device?
        # Normalization of intrinsic reward
        if reward_normalization:
            self.reward_normalizer = RewardNormalizer(reward_dim=[], until=1.0e8).to(self.device)
        else:
            self.reward_normalizer = torch.nn.Identity()

        
    def get_intrinsic_reward(self, obs) -> torch.Tensor:
        # Normalize observation, if specified
        obs_normalized = self.obs_normalizer(obs)
        # Obtain the embedding of the rnd state from the target and predictor networks
        target_embedding = self.target(obs_normalized).detach()
        predictor_embedding = self.predictor(obs_normalized).detach()
        # Compute the intrinsic reward as the distance between the embeddings
        intrinsic_reward = torch.linalg.norm(target_embedding - predictor_embedding, dim=1)
        # Normalize intrinsic reward and then rescale
        intrinsic_reward = self.reward_normalizer(intrinsic_reward) * self.reward_scale

        return intrinsic_reward
    

    def update_normalization(self, obs):
        # Normalize the state
        # reward_normalization is updated in forward()
        if self.obs_normalization:
            self.obs_normalizer.update(obs)


    def train(self, mode:bool=True)->None:
        if mode:
            self.predictor.train()
            self.obs_normalizer.train()
            self.reward_normalizer.train()
        else:
            self.predictor.eval()
            self.obs_normalizer.eval()
            self.reward_normalizer.eval()