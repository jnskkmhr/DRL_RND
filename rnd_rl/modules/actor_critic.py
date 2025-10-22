from typing import Literal
import torch 
import torch.nn as nn
from torch.distributions import Normal
from rnd_rl.modules.normalizer import ObsNormalizer

class ActorCritic(nn.Module):
    def __init__(
        self, 
        obs_dim, 
        action_dim, 
        actor_hidden_dims=[128, 128], 
        critic_hidden_dims=[128, 128],
        init_noise_std=1.0, 
        noise_std_type: Literal["scalar", "log"] = "scalar",
        device=torch.device("cpu"), 
        obs_normalization: bool = False,
        ):
        super(ActorCritic, self).__init__()
        
        self.device = device
        self.noise_std_type = noise_std_type
        
        actor_hidden_dims_processed = [obs_dim] + actor_hidden_dims + [action_dim]
        critic_hidden_dims_processed = [obs_dim] + critic_hidden_dims + [1]
        
        actor_layers = []
        critic_layers = []
        for layer_index in range(len(actor_hidden_dims_processed) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims_processed[layer_index], actor_hidden_dims_processed[layer_index + 1]))
            if layer_index < len(actor_hidden_dims_processed) - 2:
                actor_layers.append(nn.ReLU())
        for layer_index in range(len(critic_hidden_dims_processed) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims_processed[layer_index], critic_hidden_dims_processed[layer_index + 1]))
            if layer_index < len(critic_hidden_dims_processed) - 2:
                critic_layers.append(nn.ReLU())
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(action_dim)))
        
        self.distribution = None

        # normalization, if specified
        # might not really need 2 normalizers
        # since they are only updated when update() is called in each timestep.
        # and in this implementation, actor and critic obs are the same.
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = ObsNormalizer(obs_dim=[obs_dim], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity() # to device?
        
    def act(self, obs:torch.Tensor)->torch.Tensor:
        # normalize obs here
        obs_normalized = self.obs_normalizer(obs)
        self.update_distribution(obs_normalized)
        # self.update_distribution(obs)
        return self.distribution.sample()
    
    def act_inference(self, obs:torch.Tensor)->torch.Tensor:
        # normalize obs here
        obs_normalized = self.obs_normalizer(obs)
        # return self.actor(obs)
        return self.actor(obs_normalized)
    
    def evaluate(self, obs:torch.Tensor)->torch.Tensor:
        # normalize obs here
        obs_normalized = self.obs_normalizer(obs)
        # return self.critic(obs)
        return self.critic(obs_normalized)

    def update_distribution(self, obs_normalized:torch.Tensor)->None:
        # obs should be already normalized here
        mean = self.actor(obs_normalized)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)
        
    def update_normalization(self, obs):
        if self.obs_normalization:
            self.obs_normalizer.update(obs)

    """
    properties.
    """
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev