import torch
import torch.nn as nn
import torch.nn.functional as F
from rnd_rl.modules.normalizer import ObsNormalizer, RewardNormalizer

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class RandomNetworkDistillation(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [128, 128],
        device: torch.device = torch.device("cpu"),
        obs_normalization: bool = False,
        reward_normalization: bool = False,
        activation: str = "relu",  # "relu", "leaky_relu", "elu", "swish"
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super(RandomNetworkDistillation, self).__init__()
        
        self.device = device
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        
        # Define activation function
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(negative_slope=0.01)
        elif activation == "elu":
            self.activation_fn = nn.ELU()
        elif activation == "swish":
            self.activation_fn = Swish()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Build the target network (fixed)
        target_layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            target_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_batch_norm:
                    target_layers.append(nn.BatchNorm1d(dims[i + 1]))
                elif use_layer_norm:
                    target_layers.append(nn.LayerNorm(dims[i + 1]))
                target_layers.append(self.activation_fn)
        self.target = nn.Sequential(*target_layers).to(self.device)
        
        # Build the predictor network (trainable)
        predictor_layers = []
        for i in range(len(dims) - 1):
            predictor_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_batch_norm:
                    predictor_layers.append(nn.BatchNorm1d(dims[i + 1]))
                elif use_layer_norm:
                    predictor_layers.append(nn.LayerNorm(dims[i + 1]))
                predictor_layers.append(self.activation_fn)
                if dropout_rate > 0:
                    predictor_layers.append(nn.Dropout(dropout_rate))
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

        # # Obtain the embedding of the rnd state from the target and predictor networks
        # target_embedding = self.target(obs).detach()
        # predictor_embedding = self.predictor(obs).detach()
        # Compute the intrinsic reward as the distance between the embeddings
        intrinsic_reward = torch.linalg.norm(target_embedding - predictor_embedding, dim=1)
        # # Normalize intrinsic reward, already in batches?
        intrinsic_reward = self.reward_normalizer(intrinsic_reward)

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