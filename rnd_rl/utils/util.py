import random 
import numpy as np
import torch 

def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, and pytorch.
    
    Args:
        seed: Random seed value
    """
    # Python random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For even more reproducibility (may impact performance)
    # torch.use_deterministic_algorithms(True)