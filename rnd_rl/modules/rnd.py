import torch
import torch.nn as nn

class RandomNetworkDistillation(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [128, 128],
        device: torch.device = torch.device("cpu"),
    ):
        super(RandomNetworkDistillation, self).__init__()
        
        self.device = device
        
        # Build the target network (fixed)
        target_layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            target_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                target_layers.append(nn.ReLU())
        self.targe = nn.Sequential(*target_layers).to(self.device)
        
        # Build the predictor network (trainable)
        predictor_layers = []
        for i in range(len(dims) - 1):
            predictor_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                predictor_layers.append(nn.ReLU())
        self.predictor = nn.Sequential(*predictor_layers).to(self.device)
        
        # make target network not trainable
        self.target.eval()
        
    def get_intrinsic_reward(self, obs) -> torch.Tensor:
        # Obtain the embedding of the rnd state from the target and predictor networks
        target_embedding = self.target(obs).detach()
        predictor_embedding = self.predictor(obs).detach()
        # Compute the intrinsic reward as the distance between the embeddings
        intrinsic_reward = torch.linalg.norm(target_embedding - predictor_embedding, dim=1)
        # # Normalize intrinsic reward
        # intrinsic_reward = self.reward_normalizer(intrinsic_reward)

        return intrinsic_reward
    
    def train(self, mode:bool=True)->None:
        if mode:
            self.predictor.train()
        else:
            self.predictor.eval()