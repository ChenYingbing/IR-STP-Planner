import abc
from typing import Dict, Union, Any
import os
import copy

import utils.file_io

class DatasetBasicIO:
  '''
  Class to uniform the data extractor in different dataset
  '''
  def __init__(self, file_end_str: str):
    '''
    :param file_end_str: file end string when writing files
    '''
    # batch cases data format:
    #   'case_name': {
    #     case_dict_data
    #   }
    self.batch_cases_data: Dict = {}
    # case dict data format:
    #   'key_name': {
    #     data: e.g., List, Dict, ...
    #   }
    self.case_dict_data: Dict = {}

    self.file_end_str = file_end_str

  def reinit_batch_cases_data(self):
    self.batch_cases_data.clear()   # clear the data

  def reinit_case_data(self):
    self.case_dict_data.clear() # clear the data
    self._reinit_case_data()    # sub-class reinit

  ###########################################################
  # batch cases data operation
  def set_case_data2batch_cases_data(self, case_idx: int):
    self.batch_cases_data['case_{}'.format(case_idx)] =\
       copy.copy(self.case_dict_data) # @note need to use copy.

  def try_write_batch_cases_data(self, write_folder: str,
                                 scneario_idx: int, case_idx: int, 
                                 batch_num: int,
                                 forcibly_write: bool = False):
    '''
    Check if can write the batch_cases_data
    :param write_folder: path to write data
    :param batch_num: condition of batch_num to write the data
    '''
    if (len(self.batch_cases_data.keys()) >= batch_num) or \
       ((len(self.batch_cases_data.keys()) > 0) and forcibly_write):
      filename = "scenario{}_case{}_batch{}".format(
        scneario_idx, case_idx, batch_num) + self.file_end_str
      filepath = os.path.join(write_folder, filename)

      # print("write_batch_cases_data")
      # for case_str, dt in self.batch_cases_data.items():
      #   print(case_str, dt.keys())
      #   for key, ddt in dt.items():
      #     print(key, len(ddt))
      utils.file_io.write_dict2bin(
        self.batch_cases_data, filepath, verbose=False)

      self.reinit_batch_cases_data()

  ###########################################################
  # case data operation
  @abc.abstractmethod
  def _reinit_case_data(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def append_case_data(self, **kwargs):
    raise NotImplementedError()

  ###########################################################
  # read method
  @staticmethod
  def read_data(read_dict_data: Dict) -> Dict[str, Any]:
    raise NotImplementedError() 
