from pickle import TRUE
import numpy as np
from typing import Dict, List, Union, Tuple
import math

from utils.transform import XYYawTransform

class SpaceXd:
  def __init__(self, origins: List[float],
                     space_ranges: List, 
                     space_resos: List[float],
                     allocate_grid_space: bool) -> None:
    '''
    Describe search space for planner
    :param origins: origin the search space at each axis
    :param space_ranges: [[relative_min, relative_max], ...]
    :param space_resos: resolutions of the space at each axis
    :param init_grid_space: whether allocate the numpy array previously
    '''
    self.dim: int = len(origins)
    assert self.dim > 0, "SpaceXd, Dim == 0"
    assert self.dim == len(space_ranges), "SpaceXd, Dim unmatched={}.".format(self.dim, len(space_ranges))
    assert self.dim == len(space_resos), "SpaceXd, Reso unmatched={}.".format(self.dim, len(space_resos))
    self.origins = np.array(origins)
    
    self.space_ranges = space_ranges
    self.space_resos = np.array(space_resos)
    self.space_resos_2 = self.space_resos * 0.5

    self.space_extents = []
    self.grid_dimension = []
    self.grid_num: int = 1
    for i, ranges in enumerate(self.space_ranges):
      axis_extent = ranges[1] - ranges[0]
      num: int = math.ceil(axis_extent / self.space_resos[i])

      self.space_extents.append(axis_extent)
      self.grid_dimension.append(num)
      self.grid_num *= num

    # space extents: space range at each axis
    self.space_extents = np.array(self.space_extents)
    # space ranges: space ranges at each axis [min, max]
    self.space_ranges = np.array(self.space_ranges)
    # space grid dimension
    self.grid_dimension = np.array(self.grid_dimension, dtype=int)
    
    # space grids stored at numpy array
    self.allocate_grid_space = allocate_grid_space
    if allocate_grid_space:
      self.grids = np.zeros(tuple(self.grid_dimension))
    else:
      self.grids = None
  
    # space information
    #   map_grids2infos: map from grid_indexs to information_index
    # @note the information should be implemented in outsied class
    self.map_grids2infos: Dict[Tuple, set] = {}

  def reinit(self, origins: List[float]=None) -> None:
    '''
    Reinit this class
    '''
    if origins != None:
      new_dim = len(origins)
      assert new_dim == self.dim, "Error, new origin does not meet the dimension, {}.".format(
        new_dim, self.dim)
      self.origins = np.array(origins)

    if self.allocate_grid_space:
      self.grids[:] = 0.0
    self.map_grids2infos.clear()

  def print_debug_string(self) -> None:
    '''
    Print string for debug
    '''
    print(">"*30)
    print("SpaceXd::Info()")
    print("Dim={}, {}.".format(self.dim, self.grid_dimension))
    print("Ranges={}.".format(self.space_ranges))
    print("Extents={}.".format(self.space_extents))
    print("Grid orign={}.".format(self.origins))
    print("Grid reso={}.".format(self.space_resos))
    if self.allocate_grid_space:
      print("Grid shape={}.".format(self.grids.shape))
    print("<"*30)

  def is_within_grid_range(self, input: List[float]) -> bool:
    '''
    Return true if input is within grid range
    '''
    assert len(input) == self.dim, "Inputs dimension unequal: {}.".format(
      [len(input), self.dim])
    rinput = np.array(input) - self.origins + self.space_resos_2 + 1e-6

    sum_result = np.sum(
      np.logical_or(rinput < self.space_ranges[:, 0], rinput > self.space_ranges[:, 1]))
    
    # print("rinput", input, rinput)
    is_within = (sum_result < 1e-6)

    return is_within

  def _get_index(self, input: List[float]) -> Tuple:
    '''
    Return index value of each axis, do not check whether it is legal or not.
    '''
    assert len(input) == self.dim, "Inputs dimension unequal: {}.".format(
      [len(input), self.dim])
    input = np.array(input)

    result = np.floor(
      (input - self.origins - self.space_ranges[:, 0] + self.space_resos_2) / self.space_resos)

    return tuple(np.array(result, dtype=int).tolist())

  def get_index(self, input: List[float]) -> Tuple:
    '''
    Return Tuple[flag, indexs], where flag indicates whether input 
      of xyz are inside the grid range, and indexs is the corresponding grid indexes.
    '''
    if self.is_within_grid_range(input):
      return True, self._get_index(input)
    
    return False, None

  def get_grid_value(self, input: List[float]):
    '''
    Return value of a grid at input, where flag indicates whether input 
      of xyz are inside the grid range, and indexs is the corresponding grid indexes.
    '''
    if not self.allocate_grid_space:
      return False, None

    if self.is_within_grid_range(input):
      return True, self.grids[self._get_index(input)]
    
    return False, None

  def get_indexs(self, inputs: np.ndarray) -> Tuple:
    '''
    Return indexs of inputs, and the valid positions
    : return: indexs, valid_indicators
    '''
    indexs = np.array(
      np.floor(
        (inputs - self.origins - self.space_ranges[:, 0] + self.space_resos_2) / self.space_resos
        ), int)

    checks = np.sum(
        np.logical_or(indexs < 0, indexs >= self.grid_dimension), axis=1)
    valids = checks < 1e-3

    return indexs, valids

  def record_information(self, input: List[float], info_idx: int) -> None:
    '''
    Add information records to the input
    '''
    flag, tuple_index = self.get_index(input)
    if flag == True:
      if not tuple_index in self.map_grids2infos:
        self.map_grids2infos[tuple_index] = set()
      self.map_grids2infos[tuple_index].add(info_idx)

  def record_informations(self, inputs: np.ndarray, info_idx: int) -> None:
    '''
    Add information records to the space given numbers of inputs
    :param inputs: values in the format of (num, self.dim)
    :param info_idx: information index
    '''
    assert inputs.shape[1] == self.dim, "Inputs dimension unequal: {}.".format(
      [inputs.shape, self.dim])
    
    indexs, valids = self.get_indexs(inputs)
    # print("record_informations", inputs.shape, self.origins.shape)

    # is within range, map to info_idx
    for index in indexs[valids]:
      tuple_index = tuple(index)
      if not tuple_index in self.map_grids2infos:
        self.map_grids2infos[tuple_index] = set()
      self.map_grids2infos[tuple_index].add(info_idx)

  def get_grids_with_info(self, output_indexs=False) -> np.ndarray:
    '''
    Return array of indexs that has the information
    : param output_indexs: when it is true, return indexs array (int), else return grid locaitons (float)
    '''
    get_indexs = np.array([]).reshape(0, self.dim)
    for index in self.map_grids2infos.keys():
      get_indexs = np.concatenate(
        (get_indexs, np.array([index], dtype=int)), axis=0)

    if output_indexs:
      return np.array(get_indexs, dtype=int)
    else:
      # !!!not - self.space_resos_2, will cause index shift problem if retransformed to indexs
      return (get_indexs * self.space_resos) + self.origins + self.space_ranges[:, 0]
