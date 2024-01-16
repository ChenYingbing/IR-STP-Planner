import os
import torch
import numpy as np
from typing import List, Dict

from utils.transform import XYYawTransform
from models.prediction.model import BasicPredictionModel
from train_eval.trajectory_prediction.initialization import initialize_prediction_network
import train_eval.utils as train_eval_u


class PredictionNetworkModel(BasicPredictionModel):
  def __init__(self, device:str, 
                     cfg: Dict,
                     model_path: str, 
                     predict_horizon_L: int,
                     prediction_mode: str = 'default',
                     predict_traj_mode_num: int = 10,
                     thres_traj_sum_prob: float = 0.95,
                     thres_traj_min_prob: float = 0.025):
    '''
    :param predict_horizon_L: prediction horizon length
    :param prediction_mode: mode of prediction module
    :param predict_traj_mode_num: the mode amount the trajectories being picked up
    :param thres_traj_sum_prob: remove trajs if sum of trajectory probabilities is >= thres_traj_sum_prob
    :param thres_traj_min_prob: remove trajs if its probabilities < thres_traj_min_prob
    '''
    BasicPredictionModel.__init__(self, device, cfg, model_path)

    valid_modes = ['default', 'lon-short', 'lon-short-v2']
    assert prediction_mode in valid_modes, 'invalid mode of prediction, with value={} / {}.'.format(
      prediction_mode, valid_modes)

    self.predict_horizon_L = predict_horizon_L
    self.prediction_mode = prediction_mode
    self.predict_traj_mode_num = predict_traj_mode_num
    self.thres_traj_sum_prob = thres_traj_sum_prob
    self.thres_traj_min_prob = thres_traj_min_prob

    self.__is_lon_short_mode = (self.prediction_mode == 'lon-short')
    self.__short_horizon_T = 4 # 4 = 2.0s
    self.__short_horizon_T_v2 = 3.0 # 2.0s

    self.__short_horizon_D = 12.0

    self.__is_lon_short_mode_v2 = (self.prediction_mode == 'lon-short-v2')    

    self.netmodel = initialize_prediction_network(
      cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
      cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'])
    self.netmodel = self.netmodel.float().to(device)
    checkpoint = torch.load(model_path)
    self.netmodel.load_state_dict(checkpoint['model_state_dict'])
    self.netmodel.eval()

  def predict(self, ego_xyyaw: XYYawTransform,
                    agent_list: List[Dict],
                    agent_states: List[XYYawTransform],
                    inputs: List) -> List:
    '''
    Return list of agent predicted trajectories.
    '''
    assert len(agent_states) == len(inputs), "Fatal Error."

    agents_trajs = []
    _idx = 0
    for agent_info, agent_state, get_input in zip(agent_list, agent_states, inputs):
      # pred keys dict_keys(['idx', 'length', 'width', 'shape', 'xyyaw', 'velocity', 'acceleration'])
      agent_idx = agent_info['idx']
      torch_input = train_eval_u.convert2tensors(get_input)
      torch_input = train_eval_u.send_to_device(
        train_eval_u.convert_double_to_float(torch_input))

      agent_v = agent_info['velocity']
      dist2agent = ego_xyyaw.distance2(agent_state)
      is_far2ego :bool= dist2agent > max(agent_v * self.__short_horizon_T_v2, self.__short_horizon_D)

      get_output = self.netmodel(torch_input)
      # print(_idx, get_output['traj'].shape, get_output['probs'].shape)
      # >>          torch.Size([1, mode_num, 12, 2]) torch.Size([1, mode_num])

      probs = torch.softmax(get_output['probs'], dim=1)
      # print("probs", probs.shape)

      # from probability high to low
      batch_indexs =\
        probs.argsort(descending=True)[:, 0:self.predict_traj_mode_num].cpu().detach().numpy()[0]

      sum_prob = 0.0
      agent_potential_trajs = []
      is_first_pred = True
      for tid in batch_indexs: # from probability high to low
        prob = probs[0, tid].cpu().detach().item()
        if prob < self.thres_traj_min_prob:
          continue
        if sum_prob >= self.thres_traj_sum_prob:
          break

        rela_pred_traj = get_output['traj'][0, tid, :, :].detach().cpu().numpy()
        pred_traj = self.transform2global_frame(agent_state, rela_pred_traj, True)
        pred_traj = self._estimate_trajectory_yaws(agent_state, pred_traj)

        # if (agent_idx == 30015):
        #   print(agent_idx, "traj=", pred_traj)

        not_skip_this_pred = True
        if self.__is_lon_short_mode:
          if is_first_pred: # first prediction with highest probability
            # probability max prediction with full prediction horizon
            pred_traj = pred_traj[:(self.predict_horizon_L + 1), :]
          else:
            # others with limited prediction horizon
            pred_traj = pred_traj[:(self.__short_horizon_T + 1), :]
          is_first_pred = False
        elif self.__is_lon_short_mode_v2:
          pred_traj = pred_traj[:(self.predict_horizon_L + 1), :]
          if (is_far2ego == False) and (is_first_pred == False):
            not_skip_this_pred = False # skip when not far and is not first prediction result (max probability)

          is_first_pred = False
        else:
          # default situations, all prediction results with full horizon
          pred_traj = pred_traj[:(self.predict_horizon_L + 1), :] # limit the prediction horizon
        # print("pred_traj", tid, pred_traj.shape)

        if not_skip_this_pred:
          agent_potential_trajs.append(
            {'agent_idx': agent_idx, 'prob': prob, 'trajectory': pred_traj})
          sum_prob += prob

      # reweighted the probs
      for trajinfo in agent_potential_trajs:
        trajinfo['prob'] /= sum_prob

      agents_trajs.append(agent_potential_trajs)

      _idx += 1

    # print("agents_trajs", agents_trajs)
    return agents_trajs
