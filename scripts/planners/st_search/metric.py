import abc
import numpy as np
import math
from typing import Union
from planners.st_search.st_node import StNode

class CostMetric:
  def __init__(self, metric_name: str) -> None:
    self.metric_name = metric_name

  def name(self) -> str:
    return self.metric_name

  @abc.abstractmethod
  def get_action_cost(self, parent_node: StNode, child_node: StNode) -> float:
    '''
    Return action cost value given parent node and its child node
    '''
    raise NotImplementedError("CostMetric::get_action_cost() not implemented.") 

  @abc.abstractmethod
  def get_control_effort(self, acc_value: Union[float, np.ndarray], t_dur: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Return control effort
    :param acc_value: the input acc value
    :param t_dur: the duration of acc
    '''
    raise NotImplementedError("CostMetric::get_control_effort() not implemented.")

  @abc.abstractmethod
  def get_g_cost(self, parent_node: StNode, child_node: StNode) -> float:
    '''
    Return g cost value given parent node and its child node
    '''
    raise NotImplementedError("CostMetric::get_g_cost() not implemented.")

  @abc.abstractmethod
  def get_heuritic_value(self, node: StNode) -> float:
    '''
    Return heustric value for the given node
    :note: a heuristic function is said to be admissible if it never overestimates 
           the cost of reaching the goal, i.e. the cost it estimates to reach the 
           goal is not higher than the lowest possible cost from the current point in the path.
    '''
    raise NotImplementedError("CostMetric::get_heuritic_value() not implemented.") 
