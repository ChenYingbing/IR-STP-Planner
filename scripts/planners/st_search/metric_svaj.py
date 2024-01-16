from planners.st_search.metric import CostMetric
from planners.st_search.sti_node import StiNode

import numpy as np
from typing import Tuple, Union
import math

class SVAJCostMetric(CostMetric):
  def __init__(self, start_s: float, goal_s: float, 
                     speed_limits: np.ndarray,
                     svaj_weights: Tuple[float, float, float, float],
                     heuristic_coef: float = 1.0) -> None:
    '''
    Cost metric considerting longitudinal sum distance (s), speed, acceleration, and jerk
    :param start_s: s value of the start (temporarily useless)
    :param goal_s: s value of the goal
    :param speed_limits: speed limits at each s value
    :param svaj_weights: cost weights
    :param heuristic_coef: weight of heuristic function, when < 1.0, is to encourage depth-first
    '''
    CostMetric.__init__(self, metric_name="sva_metric")
    
    self.goal_s = goal_s
    self.norm_s = 1.0 # max(goal_s - start_s, minimum_norm_s)
    self.speed_limits = speed_limits

    self.svaj_weights = svaj_weights
    self.heuristic_coef = heuristic_coef

    format_node = StiNode(izone_num=0)
    self.key_pidx = format_node.get_key_index('parent_index')
    self.key_sidx = format_node.get_key_index('s_index')
    self.key_nidx = format_node.get_key_index('node_index')
    self.key_s = format_node.get_key_index('state_s')
    self.key_t = format_node.get_key_index('state_t')
    self.key_v = format_node.get_key_index('state_v')
    self.key_acc = format_node.get_key_index('state_acc')
    self.key_flag = format_node.get_key_index('leaf_flag')

  def get_action_cost(self, parent_node: StiNode, child_node: StiNode) -> float:
    '''
    Return action cost value given parent node and its child node
    '''
    delta_t = max(
      math.fabs(child_node.get_state_value('state_t') - parent_node.get_state_value('state_t')), 1e-3)

    move_s = (child_node.get_state_value('state_s') - parent_node.get_state_value('state_s'))
    cost_v = math.fabs(self.speed_limits[child_node.s_index()] - child_node.get_state_value('state_v')) \
              * delta_t # (v_u - dv) * dt
    cost_acc = child_node.get_state_value('state_acc') * child_node.get_state_value('state_acc') * delta_t             # acc*2 * dt
    cost_jerk = ((child_node.get_state_value('state_acc') - parent_node.get_state_value('state_acc')) ** 2) / delta_t  # j^2 * dt

    sum_cost = self.svaj_weights[0] * move_s +\
               self.svaj_weights[1] * cost_v +\
               self.svaj_weights[2] * cost_acc +\
               self.svaj_weights[3] * cost_jerk

    return sum_cost

  def get_action_costs(self, parent_node: np.array, child_nodes: np.array, child_v_limit: float) -> np.array:
    '''
    Return action cost values given parent node and child nodes
    '''
    delta_t = child_nodes[:, self.key_t] - parent_node[self.key_t]
    delta_t[delta_t < 1e-3] = 1e-3
    
    move_s = child_nodes[:, self.key_s] - parent_node[self.key_s]
    cost_v = np.fabs(child_v_limit - child_nodes[:, self.key_v]) / child_v_limit * delta_t # (v_u - dv) * dt
    cost_acc = np.square(child_nodes[:, self.key_acc]) * delta_t
    cost_jerk = np.square(child_nodes[:, self.key_acc] - parent_node[self.key_acc]) / delta_t

    sum_costs = self.svaj_weights[0] * move_s +\
                self.svaj_weights[1] * cost_v +\
                self.svaj_weights[2] * cost_acc +\
                self.svaj_weights[3] * cost_jerk
    return sum_costs

  def get_control_effort(self, acc_value: Union[float, np.ndarray], t_dur: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Return control effort
    :param acc_value: the input acc value
    :param t_dur: the duration of acc
    '''
    if isinstance(acc_value, float):
      return self.svaj_weights[2] * (acc_value * acc_value * t_dur)

    if isinstance(acc_value, np.ndarray):
      costs = np.square(acc_value) * t_dur
      return self.svaj_weights[2] * np.sum(costs)

    raise ValueError(
      "SVAJCostMetric::get_control_effort(), type of acc_value {} is not implemented.".format(type(acc_value)))

  def get_g_cost(self, parent_node: StiNode, child_node: StiNode) -> float:
    '''
    Return svaj g cost (action cost) given parent node and its child node
    '''
    assert child_node.get_state_value('state_s') > parent_node.get_state_value('state_s'), "Error input state_s = {}".format(
      [child_node.get_state_value('state_s'), parent_node.get_state_value('state_s')])
    assert child_node.get_state_value('state_t') >= parent_node.get_state_value('state_t'), "Error input state_t = {}".format(
      [child_node.get_state_value('state_t'), parent_node.get_state_value('state_t')])

    move_s = (child_node.get_state_value('state_s') - parent_node.get_state_value('state_s'))

    sum_cost = self.svaj_weights[0] * move_s / self.norm_s +\
               self.get_action_cost(parent_node, child_node)

    return sum_cost

  def get_heuritic_value(self, node: StiNode) -> float:
    '''
    Return heustric value for the given node
    :note: a heuristic function is said to be admissible if it never overestimates 
           the cost of reaching the goal, i.e. the cost it estimates to reach the 
           goal is not higher than the lowest possible cost from the current point in the path.
    '''

    return self.heuristic_coef * self.svaj_weights[0] *\
      (self.goal_s - node.get_state_value('state_s')) / self.norm_s
