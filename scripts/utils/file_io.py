
import os
import pickle
import yaml
from typing import Dict, List

from utils.colored_print import ColorPrinter

def extract_folder_file_list(folder_dir:str) -> List:
  file_list = []
  if os.path.isdir(folder_dir):
    file_list = os.listdir(folder_dir)
  else:
    raise ValueError("{} is not a directory.".format(folder_dir))
  
  return file_list

def write_dict2bin(dict_data, file, verbose: bool=True):
  f = open(file, 'wb')
  pickle.dump(dict_data, f)
  f.close()
  if verbose:
    ColorPrinter.print('yellow', "file writed at {}".format(file))

def read_dict_from_bin(file, verbose: bool=True):
  f = open(file, 'rb')
  get_dict = pickle.load(f)
  if verbose:
    ColorPrinter.print('green', "file readed given {}".format(file))
  return get_dict

def read_yaml_config(file_path) -> Dict:
  cfg: Dict = {}
  try:
    with open(file_path, 'r') as yaml_file:
      cfg = yaml.safe_load(yaml_file)
  except Exception as einfo:
    ColorPrinter.print('red', "Fail to read config from directory: {}.".format(file_path))

  return cfg
