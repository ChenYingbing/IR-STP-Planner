import math
import os
import abc
import pickle
from typing import Dict, Any
import torch
import torch.utils.data as torch_data
import random

from utils.file_io import extract_folder_file_list
from utils.file_io import read_yaml_config
from preprocessor.utils import InterfaceMode

class CommonroadDataset(torch_data.Dataset):
  def __init__(self, data_path: str,
                     config_path: str=None,
                     encoder_type: str=None,
                     save_path: str=None,
                     file_data_num: int=20):
    '''
    Interface to read commonroad dataset data (xml scenario files)
    :param data_path: path to scenario file list
    :param config_path: path to read config file
    :param save_path: data_path to save results
    :param file_data_num: number to data stored in one file
    '''
    self.mode: InterfaceMode = InterfaceMode.EXTRACT
    self.args_dict = {}
    if config_path:
      self.args_dict = read_yaml_config(config_path)['commonroad']

      MODE_MAP = {'extract': InterfaceMode.EXTRACT, 'load': InterfaceMode.LOAD }
      self.mode = MODE_MAP[self.args_dict['mode']]

    self.save_path = save_path
    if self.is_extract_mode():
      self.data_path = data_path
      self.scenario_list = extract_folder_file_list(data_path)
      self.file_data_num = file_data_num
    else:
      data_percentage = self.args_dict['full_loaded']
      self.version = self.args_dict['version']

      file_list = extract_folder_file_list(self.save_path)
      self.read_file_list = []
      for fname in file_list:
        if self.version in fname and encoder_type in fname and '.pickle' in fname:
          self.read_file_list.append(fname)
      ori_file_num = len(self.read_file_list)
      new_file_num = math.ceil(ori_file_num * data_percentage)

      self.read_file_list = random.sample(self.read_file_list, new_file_num)
      print("{} number ({:.1f}%, ori={}) of files being readed.".format(
        len(self.read_file_list), (data_percentage * 100.0), ori_file_num
        )
      )

  def is_extract_mode(self) -> bool:
    return (self.mode == InterfaceMode.EXTRACT)

  def __len__(self) -> int:
    if self.is_extract_mode():
      return len(self.scenario_list)
    else:
      return len(self.read_file_list)

  def __getitem__(self, idx):
    '''
    :param idx: index in scenario_list
    '''
    if self.is_extract_mode():
      scene_name = self.scenario_list[idx]
      scene_path = os.path.join(self.data_path, scene_name)

      get_data = self.process_data(
        idx, scene_name, self.file_data_num, scene_path=scene_path)

      return 0  
    else:
      fname = self.read_file_list[idx]
      get_data = self.load_data(fname)
      # print("read file name= {}.".format(fname))
      return self.read_data_from_file(idx, get_data)

  @abc.abstractmethod
  def process_data(self, idx: int, scenario_name: str,
                         file_data_num: int, **kwargs):
    '''
    Process data given index of map.csv
    :param idx: index of map.csv in dataset folder
    '''
    raise NotImplementedError()

  @abc.abstractmethod
  def read_data_from_file(self, idx: int, file_data: Dict):
    '''
    Read data given data readed from the file
    :param file_data: the data readed from the file
    '''
    raise NotImplementedError()

  def save_data(self, filename: str, data: Dict):
      filepath = os.path.join(self.save_path, filename)
      with open(filepath, 'wb') as handle:
          pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def load_data(self, filename: str) -> Dict:
      """
      Function to load extracted data.
      :param idx: data index
      :return data: Dictionary with pre-processed data
      """
      filepath = os.path.join(self.save_path, filename)
      if not os.path.isfile(filepath):
          raise Exception('Could not find data. Please run the dataset in extract_data mode')

      with open(filepath, 'rb') as handle:
          data = pickle.load(handle)
      return data

