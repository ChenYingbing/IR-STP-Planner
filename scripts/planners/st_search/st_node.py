import numpy as np
from typing import List

class StNode:
  def __init__(self) -> None:
    self.parent_node_idx: int= -1

    self.s_sample_idx: int= 0
    self.node_idx: int= -1

    self.state_s: float= 0.0
    self.state_t: float= 0.0
    self.state_v: float= 0.0
    self.state_acc: float= 0.0

    # flag being set to indicate this node is leaf node
    self.leaf_node_flag: bool= False

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

  def get_key_index(self, key_str: str) -> int:
    '''
    Return index of values in the array
    '''
    return self.__key_map[key_str]

  def update_values(self, input_array: np.ndarray) -> None:
    '''
    Update class members given array-like input
    '''
    self.parent_node_idx = int(input_array[0])
    self.s_sample_idx = int(input_array[1])
    self.node_idx = int(input_array[2])
    self.state_s = input_array[3]
    self.state_t = input_array[4]
    self.state_v = input_array[5]
    self.state_acc = input_array[6]
    self.leaf_node_flag = (input_array[7] > 0.5)

  @staticmethod
  def len_list_values() -> int:
    '''
    Return length of the data of get_list_of_values()
    '''
    return 8

  def get_list_of_values(self) -> List:
    '''
    Return list of values
    '''
    return [self.parent_node_idx, 
            self.s_sample_idx, 
            self.node_idx,
            self.state_s,
            self.state_t,
            self.state_v,
            self.state_acc,
            self.leaf_node_flag * 1.0]

  def get_array_of_values(self) -> np.ndarray:
    '''
    Return array like data
    '''
    return np.array(self.get_list_of_values())

  def debug_string(self) -> str:
    '''
    Return debug string
    '''
    return "[{}: stva=({:.1f}, {:.1f}, {:.1f}, {:.1f})]".format(
      self.node_idx, self.state_s, self.state_t, self.state_v, self.state_acc)
