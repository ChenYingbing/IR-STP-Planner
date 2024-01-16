import os
import random
import numpy as np
import torch

def set_random_seeds(seed: int=0, deterministic_nn: bool=False):
  os.environ["PL_GLOBAL_SEED"] = str(seed)

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  
  torch.backends.cudnn.deterministic = deterministic_nn
