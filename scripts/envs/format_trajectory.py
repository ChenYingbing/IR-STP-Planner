import os
from typing import Dict, Any

import type_utils.state_trajectory as state_traj
from envs.format import DatasetBasicIO
from .utils import estimate_trajectory_yaws

class DatasetTrajecotryIO(DatasetBasicIO):
  def __init__(self):
    super().__init__(file_end_str='.trajs.bin')

  def _reinit_case_data(self):
    '''
    Init a pack of data named case data
    case of data: means pack of data that obtained from a same scenario / timestamp data
    '''
    self.case_dict_data = {
      'local_traj': [], 'global_traj': []
    }

  def append_case_data(self, **kwargs):
    assert isinstance(kwargs['local_traj'], state_traj.StateTrajectory)
    assert isinstance(kwargs['global_traj'], state_traj.StateTrajectory)

    local_traj = kwargs['local_traj']
    global_traj = kwargs['global_traj']

    list_local_traj = local_traj.list_trajectory()
    list_global_traj = global_traj.list_trajectory()
      
    if 'correct_yaw' in kwargs:
      if kwargs['correct_yaw'] == True:
        list_local_traj = estimate_trajectory_yaws(list_local_traj)
        list_global_traj = estimate_trajectory_yaws(list_global_traj)
      
    self.case_dict_data['local_traj'].append(list_local_traj)
    self.case_dict_data['global_traj'].append(list_global_traj)

  @staticmethod
  def read_data(read_dict_data: Dict) -> Dict[str, Any]:
    '''
    Return list of trajectories.
    '''
    dict_data = {
      'local_traj': [],
      'global_traj': []
    }
    for case_str, dt in read_dict_data.items():
      # print(case_str, dt.keys())
      dict_data['local_traj'] += dt['local_traj']
      dict_data['global_traj'] += dt['global_traj']
    return dict_data
