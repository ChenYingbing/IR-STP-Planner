from typing import List, Dict, Any
import collections
import copy
import math
import numpy as np
from PIL import Image
import os
import random

import thirdparty.config
from commonroad.scenario.scenario import Scenario

from envs.commonroad.commonroad import CommonroadDataset
import type_utils.agent as agent_utils
import type_utils.state_trajectory as state_traj
from utils.transform import XYYawTransform
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.obstacle import ObstacleType
from envs.config import get_root2folder
from preprocessor.utils import InterfaceMode

from preprocessor.commonroad.vectorizer import CommonroadVectorizer
from preprocessor.commonroad.graph import CommonroadGraphProcessor

class CommonroadDatasetPredictionExtractor(CommonroadDataset):
  def __init__(self, data_path: str,
                     config_path: str=None,
                     encoder_type: str=None,
                     save_path: str=None,
                     file_data_num: int=1):
    '''
    Interface to read commonroad dataset and extract the prediction inputs/outputs
    :param data_path: path to scenario_list
    :param config_path: path to read config file
    :param save_path: path to save results
    :param file_data_num: number to data stored in one file
    ''' 
    super().__init__(data_path=data_path,
                     config_path=config_path,
                     encoder_type=encoder_type,
                     save_path=save_path,
                     file_data_num=file_data_num)
    args_dict = self.args_dict

    if self.is_extract_mode():
      self.process_map = {
        'vectorize': CommonroadVectorizer,
        'graph': CommonroadGraphProcessor,
      }
      self.process_mode = args_dict['process_mode']
      self.train_val_split = args_dict['train_val_split']
      self.t_history: float= args_dict['t_history']
      self.t_future: float= args_dict['t_future']
      self.extract_t_interval: float= args_dict['extract_t_interval']
      self.enable_rviz: bool= args_dict['enable_rviz']

      train_num = int(self.__len__() * self.train_val_split)
      eval_num = self.__len__() - train_num
      full_idxs = range(0, self.__len__())
      self.train_idxs =\
        random.sample(full_idxs, train_num)
      self.eval_idxs = []
      for idx in full_idxs:
        if not idx in self.train_idxs:
          self.eval_idxs.append(idx)

  def add_train_eval_prefix(self, idx: int, file_name: str) -> str:
    prefix_str = "eval_"
    if idx in self.train_idxs:
      prefix_str = "train_"
    return (prefix_str + file_name)

  def get_filename(self, idx: int, agent_idx: int, frame_from_idx: int) -> str:
    return self.add_train_eval_prefix(
      idx, "{}_idx[{}]_agent[{}]_frame[{}].pickle".format(
        self.process_mode, idx, agent_idx, frame_from_idx)
    )

  def get_imagepath(self, idx: int, agent_idx: int, frame_from_idx: int) -> str:
    folderpath = get_root2folder(self.save_path, 
      self.add_train_eval_prefix(idx, "{}_rviz".format(self.process_mode))
    )
    filename = "idx[{}]_agent[{}]_frame[{}].png".format(idx, agent_idx, frame_from_idx)
    
    return os.path.join(folderpath, filename)
  
  def process_data(self, idx: int, scenario_name: str, 
                         file_data_num: int, **kwargs):
    '''
    Process data given index of map.csv
    :param idx: index of map.csv in dataset folder
    :param file_data_num: useless in this case
    '''
    # check if legal
    scene_path: str = kwargs['scene_path']
    get_data = None

    is_Tmap = 'T-' in scenario_name
    if is_Tmap == True:
      scenario = None
      try:
        scenario, planning_problem_set = CommonRoadFileReader(scene_path).open()
      except Exception as einfo:
        print("Skip scenario[{}]= {}.".format(idx, scenario_name))
      if scenario == None:
        return

      print("Process scenario[{}/{}]= {}.".format(
        idx, self.__len__(), scenario_name))
      self.extract_scenario_data(idx, scenario)
    
    return get_data

  def read_data_from_file(self, idx: int, file_data: Dict):
    '''
    Read data given data readed from the file
    :param file_data: the data readed from the file
    '''
    assert len(file_data.keys()) == 1, "Error, do not support situation of key num > 1"
    case_key_list = list(file_data.keys())
    get_data = file_data[case_key_list[0]]['data']
    # print("Read idx={}".format(idx), get_data.keys())
    # print(get_data['inputs'].keys())
    return get_data

  def extract_scenario_data(self, idx: int, scenario: Scenario) -> None:
    process_tool = self.process_map[self.process_mode](self.args_dict)
    
    process_tool.init_scenario(scenario,
      enable_rviz=self.enable_rviz)

    # Compute statistics: max_frame
    extract_dframe: int= int(self.extract_t_interval / scenario.dt)
    future_frame_len: int= int(self.t_future / scenario.dt)
    # print("future_frame_len", self.t_future, future_frame_len)

    max_frame: int = 0
    frame_from_to: Dict[int, List[int]] = {}
    for _, obstacle in enumerate(scenario.obstacles):
      agent_idx = obstacle.obstacle_id
      state_list = obstacle.prediction.trajectory.state_list

      frame0: int= obstacle.initial_state.time_step
      frame1: int= frame0 + len(state_list)
      frame_from_to[agent_idx] = [frame0, frame1]

      max_frame = max(max_frame, frame1)

    # Process
    frame_from_list = range(0, max_frame, extract_dframe)
    dict_data = {}
    for frame_from in frame_from_list:
      frame_to = frame_from + future_frame_len
      process_tool.init_frame_from_to(frame_from, frame_to)

      for _, obstacle in enumerate(scenario.obstacles):
        agent_idx = obstacle.obstacle_id

        frame0: int= frame_from_to[agent_idx][0]
        frame1: int= frame_from_to[agent_idx][1]
        if (frame_from < frame0) or (frame_from >= frame1) or (frame_to >= frame1):
           # out of frame range
           # cond1+cond3: agent is valid in [frame_from, frame_to)
           # cond2: agent is valid in [frame_from, frame_to], too
           continue

        dict_data[(idx, agent_idx, frame_from)] = {
          'idx': idx,
          'agent_idx': agent_idx,
          'frame_from': frame_from,
          'data': process_tool.process(idx, agent_idx),
        }

        content = dict_data[(idx, agent_idx, frame_from)]
        if 'rviz' in content['data']:
          get_img = content['data']['rviz']
          del content['data']['rviz']

          img = Image.fromarray(get_img)
          img.save(self.get_imagepath(idx, agent_idx, frame_from))

        if len(dict_data.keys()) >= self.file_data_num:
          self.save_data(self.get_filename(idx, agent_idx, frame_from), dict_data)
          dict_data.clear()
 
        # for obstacle
      # for frame_from
    
    if len(dict_data.keys()) > 0:
      self.save_data(self.get_filename(idx, -1, -1), dict_data)
      dict_data.clear()
