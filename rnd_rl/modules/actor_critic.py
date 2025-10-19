from typing import Literal
import torch 
import torch.nn as nn
from torch.distributions import Normal

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
        
    def act(self, obs:torch.Tensor)->torch.Tensor:
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def act_inference(self, obs:torch.Tensor)->torch.Tensor:
        return self.actor(obs)
    
    def evaluate(self, obs:torch.Tensor)->torch.Tensor:
        return self.critic(obs)

    def update_distribution(self, obs:torch.Tensor)->None:
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)
        
    """
    properties.
    """
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev