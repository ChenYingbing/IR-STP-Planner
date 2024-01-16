from typing import List, Dict, Any
import collections
import copy
import math
import numpy as np

import thirdparty.config
from commonroad.scenario.scenario import Scenario

from envs.format_trajectory import DatasetTrajecotryIO

from envs.commonroad.commonroad import CommonroadDataset
import type_utils.agent as agent_utils
import type_utils.state_trajectory as state_traj
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import ObstacleType
from utils.file_io import read_yaml_config

class CommonroadDatasetTrajectoryExtractor(CommonroadDataset):
  def __init__(self, data_path: str,
                     config_path, str=None,
                     save_path: str=None,
                     file_data_num: int=20):
    '''
    Interface to read commonroad dataset and extract the trajectories
    :param data_path: path to scenario_list
    :param config_path: path to read config file
    :param save_path: path to save results
    :param file_data_num: batch number to process samples data
    ''' 
    super().__init__(data_path=data_path,
                     config_path=config_path,
                     save_path=save_path,
                     file_data_num=file_data_num)
    self.traj_extract_dt :float= self.args_dict['extract_dt']
    self.traj_extract_dur :float= self.args_dict['extract_dur']

    self.maximum_traj_length = 0.

  def process_data(self, idx: int, scenario_name: str, 
                         file_data_num: int, **kwargs):
    '''
    Process data given index of map.csv
    :param idx: index of map.csv in dataset folder
    :param file_data_num: useless in this case
    '''
    # check if legal
    scene_path: str = kwargs['scene_path']
    is_Tmap = 'T-' in scenario_name
    if is_Tmap == True:
      scenario = None
      try:
        scenario, planning_problem_set = CommonRoadFileReader(scene_path).open()
      except Exception as einfo:
        print("Skip scenario[{}]= {}.".format(idx, scenario_name))
      if scenario == None:
        return

      extractor = DatasetTrajecotryIO()
      extractor.reinit_batch_cases_data()

      print("Extract scenario[{}] {}.".format(idx, scenario_name))
      self.append_case_data(extractor, idx, scenario)
      extractor.set_case_data2batch_cases_data(idx)

      extractor.try_write_batch_cases_data(
        self.save_path, idx, 0, file_data_num, forcibly_write=True)
    
    return None

  def append_case_data(self, extractor: DatasetTrajecotryIO, 
                       idx: int, scenario: Scenario) -> None:

    extractor.reinit_case_data()

    frame_from: int=0
    for dyn_obst in scenario.dynamic_obstacles:
      traj_state_list = dyn_obst.prediction.trajectory.state_list

      agent_type_id = agent_utils.AGENT_HUMAN_INDEX if (
        dyn_obst.obstacle_type == ObstacleType.PEDESTRIAN) else\
          agent_utils.AGENT_VEHICLE_INDEX
      state0 = dyn_obst.initial_state
      frame0 = state0.time_step
      frame0_s = float(frame0) * scenario.dt
      shape = dyn_obst.obstacle_shape

      info = state_traj.TrajectoryInfo(
        scene_id=idx,
        agent_type=agent_type_id,
        length=shape.length,
        width=shape.width,
        first_time_stamp_s=frame0_s,
        time_interval_s=scenario.dt
      )

      global_traj = state_traj.StateTrajectory(info=info)
      # print("common_trajectory::debug::print()")
      # print("t0={}, len={}.".format(frame0_s, len(traj_state_list)))
      for tid, state in enumerate(traj_state_list):
        key_time_s = frame0_s + tid * scenario.dt
        state_v = state.velocity

        global_traj.append_state(
          state.position[0], state.position[1], 
          state.orientation, state_v, key_time_s)

      global_piece_trajs = global_traj.split_trajectory(
        split_dt=self.traj_extract_dt, 
        split_dur=self.traj_extract_dur)

      local_piece_trajs = []
      for gtraj in global_piece_trajs:
        ltraj = gtraj.get_local_frame_trajectory()
        local_piece_trajs.append(ltraj)
      assert len(local_piece_trajs) == len(global_piece_trajs), "Unexpected Error!"

      for ltraj, gtraj in zip(local_piece_trajs, global_piece_trajs):
        extractor.append_case_data(
          local_traj=ltraj,
          global_traj=gtraj,
          correct_yaw=agent_utils.is_human(agent_type_id)
        )
    # print(" ")
