import numpy as np
import random
import torch

def set_seed(random_seed = 42):
    random.seed(random_seed)

    # Set a random seed for NumPy
    np.random.seed(random_seed)

    # Set a random seed for PyTorch
    torch.manual_seed(random_seed)