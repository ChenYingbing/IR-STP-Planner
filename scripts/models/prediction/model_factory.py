import os
import yaml

from models.prediction.prediction_model import PredictionNetworkModel

class PredictionModelFactory:
  @staticmethod
  def produce(device:str, model_path: str, check_point: str,
              prediction_mode: str,
              predict_horizon_L: int,
              predict_traj_mode_num: int):
    '''
    Produce a prediction model
    :param prediction_mode: mode of prediction module
    :param predict_horizon_L: prediction horizon length
    :param predict_traj_mode_num: trajectory mode amount of predictions
    '''
    rmodel = None

    # Load config
    with open(os.path.join(model_path, 'config.yaml'), 'r') as yaml_file:
      cfg = yaml.safe_load(yaml_file)
      checkpoint_path = os.path.join(model_path, 'checkpoints/{}.tar'.format(check_point))

      rmodel = PredictionNetworkModel(
        device=device, cfg=cfg, model_path=checkpoint_path,
        prediction_mode=prediction_mode,
        predict_horizon_L=predict_horizon_L,
        predict_traj_mode_num=predict_traj_mode_num)

    return rmodel
