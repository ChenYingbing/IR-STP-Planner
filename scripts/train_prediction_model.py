import argparse
import yaml
from train_eval.trajectory_prediction.train_eval import TrainAndEvaluator
from torch.utils.tensorboard import SummaryWriter
import os

import envs.config
from utils.time import get_date_str

def get_model_path(dataset_name: str, model_name: str):
  '''
  Return the path to store the processed data of the corresponding dataset
  :param dataset_name: the type of dataset to store
  :param process_mode: a arbitary string to identify the mode 
  '''
  folder_name = 'models/' + model_name+'_{}'.format(get_date_str())
  path = envs.config.get_dataset_exp_folder(dataset_name, folder_name)

  return path

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
parser.add_argument("-m", "--model_name", help="name of model for identification", required=True)
parser.add_argument("-n", "--num_epochs", help="Number of epochs to run training for", required=True)
parser.add_argument("-w", "--checkpoint", help="Path to pre-trained or intermediate checkpoint", required=False)
args = parser.parse_args()

# Load config
output_dir = None
with open(args.config, 'r') as yaml_file:
  cfg = yaml.safe_load(yaml_file)
  output_dir = get_model_path(cfg['dataset'], args.model_name)

  # Make directories
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  if not os.path.isdir(os.path.join(output_dir, 'checkpoints')):
    os.mkdir(os.path.join(output_dir, 'checkpoints'))
  if not os.path.isdir(os.path.join(output_dir, 'tensorboard_logs')):
    os.mkdir(os.path.join(output_dir, 'tensorboard_logs'))

  # Save config file to model path
  with open(os.path.join(output_dir, 'config.yaml'), 'w') as wfile:
    documents = yaml.dump(cfg, wfile)

# Initialize tensorboard writer
writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard_logs'))

# Train
teval_tool = TrainAndEvaluator(cfg, checkpoint_path=args.checkpoint, writer=writer)
teval_tool.train(num_epochs=int(args.num_epochs), output_dir=output_dir)

# Close tensorboard writer
writer.close()
