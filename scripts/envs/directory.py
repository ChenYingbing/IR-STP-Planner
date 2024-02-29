import os

import envs.config
from envs.config import get_root2folder

class ProcessedDatasetDirectory:
  @staticmethod
  def get_path(dataset_type: str, process_mode: str):
    '''
    Return the path to store the processed data of the corresponding dataset
    :param dataset_type: the type of dataset to store
    :param process_mode: a arbitary string to identify the mode 
    '''
    CACHE_PATH = {
      # 'interaction_dataset':\
      #   get_root2folder(envs.config.INTERACTION_EXP_ROOT, process_mode),
      'commonroad':\
        get_root2folder(envs.config.COMMONROAD_EXP_ROOT, process_mode),
    }

    return CACHE_PATH[dataset_type]

