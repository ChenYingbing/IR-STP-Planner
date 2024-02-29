import os
import argparse
from torch.utils.data import DataLoader

import envs.config
# from envs.interaction_dataset.interaction_dataset_trajectory import InteractionDatasetTrajectoryExtractor
from envs.commonroad.commonroad_trajectory import CommonroadDatasetTrajectoryExtractor
from envs.commonroad.commonroad_prediction import CommonroadDatasetPredictionExtractor
from analysis.config import ProcessConfig
from envs.directory import ProcessedDatasetDirectory

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, default='',
                      help="determine which dataset to process")
  parser.add_argument("--process", type=str, default='trajectory',
                      help="determine what to process")
  parser.add_argument("--batch", type=int, default=1,
                      help="the number of batch to process")
  parser.add_argument("--num_workers", type=int, default=1,
                      help="the number of workers to process")
  parser.add_argument("--save_path", type=str, default=None,
                      help="path to store the processed results")
  args = parser.parse_args()

  # Preparation
  LEGAL_DATASET = ['interaction_dataset', 'commonroad']
  if not args.dataset in LEGAL_DATASET:
    raise ValueError(
      "--dataset is illegal, value should be inside {}.".format(LEGAL_DATASET))

  if not args.process in ProcessConfig.LEGAL_MODES:
    raise ValueError(
      "--process is illegal, value should be inside {}.".format(ProcessConfig.LEGAL_MODES))

  if args.save_path == None:
    args.save_path = ProcessedDatasetDirectory.get_path(args.dataset, args.process)

  # Process
  def get_dataset(dataset:str, process: str):
    PROCESSORS = {
      # 'interaction_dataset': {
      #   'trajectory': InteractionDatasetTrajectoryExtractor,
      # },
      'commonroad': {
        'trajectory': CommonroadDatasetTrajectoryExtractor,
        'prediction': CommonroadDatasetPredictionExtractor,
      },
    }
    CONFIG = {
      'trajectory': os.path.join(
        os.path.dirname(envs.__init__.__file__), 
        "conf/preprocess_trajectory.yaml"),
      'prediction': os.path.join(
        os.path.dirname(envs.__init__.__file__), 
        "conf/preprocess_dataset_config.yaml"),
    }
    FOLDER = {
      'interaction_dataset': {
        'trajectory': 'train',
      },
      'commonroad': {
        'trajectory': envs.config.get_dataset_exp_folder('commonroad', "simulated_scenarios"),
        'prediction': envs.config.get_dataset_exp_folder('commonroad', "simulated_scenarios"),
      },
    }

    return PROCESSORS[dataset][process](
      data_path=FOLDER[dataset][process],
      config_path=CONFIG[process],
      save_path=args.save_path)

  dataset = get_dataset(args.dataset, args.process)
  dataloader = DataLoader(dataset, batch_size=args.batch, 
                                   shuffle=False, 
                                   num_workers=args.num_workers)

  for i, _ in enumerate(dataloader):
    pass # processing the datas
