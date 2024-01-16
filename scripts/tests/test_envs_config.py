import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs.config
envs.config.get_dataset_exp_folder('commonroad', 'visuals')
