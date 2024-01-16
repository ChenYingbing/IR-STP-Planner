import abc
from typing import List, Dict
import torch
import numpy as np
import math

from utils.transform import XYYawTransform

class BasicPredictionModel:
  def __init__(self, device:str, cfg: str, model_path: str, predict_batch_siz: int = 5):
    self.device = device
    self.cfg = cfg
    self.model_path = model_path
    self.predict_batch_siz = predict_batch_siz

  def _estimate_trajectory_yaws(self, initial_state: XYYawTransform, ori_traj: np.ndarray, 
                                      enable_debug: bool=False) -> np.ndarray:
    '''
    Estimate yaw values of states in ori_traj, and update it.
    :param initial_state: initial state of the agent
    :param ori_traj: original prediction trajectory of the agent, with shape = [num, 3]
    '''
    nodes_num = ori_traj.shape[0]
    get_traj = ori_traj.copy()
    if nodes_num > 1:
      dxys = ori_traj[1:, :] - ori_traj[:-1, :]
      yaws = np.arctan2(dxys[:, 1], dxys[:, 0])
      # print(yaws.shape, ori_traj.shape) # shape= (11,) (12, 3)
      get_traj[1:, 2] = yaws
      get_traj[0, 2] = initial_state._yaw

      first_xy = get_traj[0, [0, 1]]
      last_xy = get_traj[-1, [0, 1]]
      dist = np.linalg.norm(last_xy - first_xy)
      if dist < 0.4: 
        # set all yaw = yaw0 when prediction traj is too short
        get_traj[:, 2] = initial_state._yaw
        # print("correct here")
      else:
        # print("correct here v2")
        # correct yaw when dist is too near
        size_1 = get_traj.shape[0] - 1
        for i, dxy in enumerate(dxys):
          ddist = math.sqrt(dxy[0]**2 + dxy[1]**2)
          if (ddist < 1e-1) and (i < size_1):
            # print(i, ddist, get_traj[i, 2], get_traj[i+1, 2])
            get_traj[i+1, 2] = get_traj[i, 2]

    elif nodes_num > 0:
      get_traj[0, 2] = initial_state._yaw

    return get_traj

  def transform2global_frame(self, agent_xyyaw: XYYawTransform,
                                   agent_rela_traj: List,
                                   enable_add_initial_state: bool = False):
      '''
      Transform the trajectory from agent frame to global frame,
      where the yaw value of states in trajectory is estimated.
      :param agent_xyyaw: transform from global frame to agent frame
      :param agent_rela_traj, list of [x, y] based on agent frame
      '''
      global_traj = [[agent_xyyaw._x, agent_xyyaw._y, agent_xyyaw._yaw]] if enable_add_initial_state == True else []
      for state in agent_rela_traj:
        gxyyaw = agent_xyyaw.multiply_from_right(
            XYYawTransform(state[0], state[1])
          )
        global_traj.append([gxyyaw._x, gxyyaw._y, 0.0])
      global_traj = np.array(global_traj) # (12, 3)

      return global_traj

  @abc.abstractmethod
  def predict(self, ego_xyyaw: XYYawTransform,
                    agent_list: List[Dict],
                    agent_states: List[XYYawTransform],
                    inputs: List) -> List:
    raise NotImplementedError()
