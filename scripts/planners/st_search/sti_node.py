import numpy as np
from typing import List, Tuple

class StiNode:
  '''
  node with s-t and interaction information
  '''
  def __init__(self, izone_num: int) -> None:
    self.__key_map = {
      'parent_index': int(0),
      's_index': int(1),
      'node_index': int(2),
      'state_s': int(3),
      'state_t': int(4),
      'state_v': int(5),
      'state_acc': int(6),
      'leaf_flag': int(7),
    }

    # __dta is with state values + relation flags
    r_bias :int= self.relation_index_bias()

    self.__dta = np.zeros(r_bias + izone_num)
    self.__izone_num = izone_num

    self.__dta[r_bias:] = self.relation_not_determined()

    # self.parent_node_idx: int= -1
    # self.s_sample_idx: int= 0
    # self.node_idx: int= -1
    # self.state_s: float= 0.0
    # self.state_t: float= 0.0
    # self.state_v: float= 0.0
    # self.state_acc: float= 0.0
    # self.leaf_node_flag: bool= False
    self.set_state_value('parent_index', -1.0)
    self.set_state_value('node_index', -1.0)

  def interaction_zone_num(self) -> int:
    '''
    Return interaction zone number
    '''
    return self.__izone_num

  def get_key_index(self, key_str: str) -> int:
    '''
    Return index of values in the array
    '''
    return self.__key_map[key_str]

  # record t-s-v values
  @staticmethod
  def tsv_record_bias() -> int:
    return 8

  @staticmethod
  def tsv_record_interval() -> int:
    return 3 # (t, s, v)

  @staticmethod
  def tsv_record_indexs() -> List:
    return [[8, 9, 10], [11, 12, 13]] # [[8, 9, 10], [11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]]

  @staticmethod
  def tsv_record_indexs_tiled() -> List:
    return [8, 9, 10, 11, 12, 13] # , 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

  @staticmethod
  def tsv_relative_idexes() -> Tuple:
    # return [0, 3, 6, 9, 12], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14]
    return [0, 3], [1, 4], [2, 5]

  # record relation indexes
  @staticmethod
  def relation_index_bias() -> int:
    return 14 # 23

  @staticmethod
  def relation_not_determined() -> float:
    return 0.0

  @staticmethod
  def relation_preempt() -> float:
    return 1.0

  @staticmethod
  def relation_yield() -> float:
    return -1.0

  # @staticmethod
  # def relation_ignored() -> float:
  #   return 0.25  

  @staticmethod
  def relation_influ() -> float:
    return 2.0

  def len_list_values(self) -> int:
    '''
    Return length of the data of get_list_of_values()
    '''
    return self.__dta.shape[0]

  def set_state_value(self, key_str:str, value: float) -> None:
    self.__dta[self.__key_map[key_str]] = value

  def parent_node_index(self) -> int:
    return int(self.__dta[self.__key_map['parent_index']])

  def s_index(self) -> int:
    return int(self.__dta[self.__key_map['s_index']])

  def node_index(self) -> int:
    return int(self.__dta[self.__key_map['node_index']])

  def is_leaf_node(self) -> bool:
    return self.__dta[self.__key_map['leaf_flag']] > 0.5

  def get_state_value(self, key_str:str) -> float:
    return self.__dta[self.__key_map[key_str]]

  def update_values(self, input_dt: np.ndarray) -> None:
    '''
    Update class members given array-like input
    '''
    self.__dta = input_dt

  def get_list_of_values(self) -> List:
    '''
    Return list of values
    '''
    return self.__dta.tolist()

  def get_array_of_values(self) -> np.ndarray:
    '''
    Return array like data
    '''
    return self.__dta

  def debug_string(self) -> str:
    '''
    Return debug string
    '''
    return "[{}: stva=({:.1f}, {:.1f}, {:.1f}, {:.1f})]".format(
      self.get_state_value('node_index'), 
      self.get_state_value('state_s'), 
      self.get_state_value('state_t'), 
      self.get_state_value('state_v'), 
      self.get_state_value('state_acc'))
