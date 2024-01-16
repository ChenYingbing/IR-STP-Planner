from typing import Dict, List, Tuple, Union
import numpy as np
import math
import copy
import abc

from planners.st_search.zone_stv_search import ZoneStvGraphSearch
from planners.interaction_space import InteractionFormat, InteractionSpace
from type_utils.agent import EgoAgent

class ZoneStvGraphSearchCollisionAvoidance(ZoneStvGraphSearch):
  '''
  Speed search based on the S-t-v graph, with collision-avoidance interaction modelling
  '''
  def __init__(self, ego_agent: EgoAgent, 
                     ispace: InteractionSpace, 
                     s_samples: np.ndarray, 
                     xyyawcur_samples: np.ndarray,
                     start_sva: Tuple[float, float, float],
                     search_horizon_T: float, 
                     planning_dt: float,
                     prediction_dt: float,
                     s_end_need_stop: bool,
                     friction_coef: float = 0.35,
                     path_s_interval: float = 1.0,
                     enable_debug_rviz: bool = False) -> None:
    '''
    :param ispace: interaction space used for cost evaluation
    :param s_samples: discretized s values of samples
    :param start_sva: initial s,v,a values of AV: corresponding to self.s_samples[0]
    :param xyyawcur_samples: [[x, y, yaw, curvature], ...] values at s_samples
    :param search_horizon_T: search(plan) horizon T
    :param planning_dt: planning node interval timestamp
    :param prediction_dt: prediction trajectory node interval timestamp
    :param s_end_need_stop: when AV reaches s_samples[-1], is need to stop or change lane in advance.
    :param friction_coef: friction coefficient to calculate speed limit for bended path
    :param path_s_interval: interval s value for s sampling
    '''
    super().__init__(
      ego_agent, ispace, s_samples, xyyawcur_samples, start_sva, search_horizon_T, 
      planning_dt, prediction_dt, s_end_need_stop, friction_coef,
      path_s_interval, enable_debug_rviz=enable_debug_rviz)

  def _edge_is_valid(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from __extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the edge's child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0

    # if (range_iinfo_dict['has_interaction']) and (child_nodes.shape[0] > 0):
    #   s_index :int= self.zone_part_data_len + InteractionFormat.iformat_index('av_s')
    #   t_index :int= self.zone_part_data_len + InteractionFormat.iformat_index('agent_t')
    #   ###################################################
    #   # Verion 0: using matrix, progress is higher but still underperform A5 settings, with highest collision rate
    #   #   A0 metric: 50.55 & 4.64\%  & 3.37 & 0.673 & 18 & 9
    #   #   A1 metric: 50.39 & 5.80\%  & 4.25 & 0.663 & 18 & 9
    #   # Note: in computation time experiment, we use this version
    #   range_pred_st_array = range_iinfo_dict['izone_data'][:, [s_index, t_index]]
    #   inquiry_s_array = range_pred_st_array[:, 0]
    #   check_stva_array = self._interpolate_nodes_given_s(
    #     parent_node, child_nodes, inquiry_s=inquiry_s_array)

    #   st_dists = range_pred_st_array - check_stva_array[:, :, :2]

    #   st_conditions = np.zeros_like(st_dists)
    #   st_conditions[:, :, 0] = self.path_protect_s
    #   st_conditions[:, :, 1] = self.yield_safe_time_gap
    #   stop_locs = check_stva_array[:, :, 2] < self.stop_v_condition
    #   st_conditions[stop_locs, 1] = 1e+3 # those stop nodes will avoid collisions at whole future time

    #   s_collisions = np.logical_and(
    #     -st_conditions[:, :, 0] <= st_dists[:, :, 0],
    #     st_dists[:, :, 0] <= st_conditions[:, :, 0]
    #   )
    #   t_collisions = np.logical_and(
    #     -self.safe_time_gap < st_dists[:, :, 1],
    #     st_dists[:, :, 1] < st_conditions[:, :, 1]
    #   )
    #   collisions = np.logical_and(s_collisions, t_collisions)

    #   for ii, collision in enumerate(collisions):
    #     if np.sum(collision) > 1e-3:
    #       valid_costs[ii, 0] = 0.0 # set 0 if collision

    # return valid_costs

    if (range_iinfo_dict['has_interaction']) and (child_nodes.shape[0] > 0):
      s_index :int= self.zone_part_data_len + InteractionFormat.iformat_index('av_s')
      t_index :int= self.zone_part_data_len + InteractionFormat.iformat_index('agent_t')
      ###################################################
      # Verion 1: using loop, progress is lesser and underperform A5 settings, with equal collision rate to A5's
      #   A0 metric: 47.92 & 6.95\%  & 4.22 & 0.628 & 15 & 9
      #   A1 metric: 45.68 & 10.16\%  & 5.82 & 0.630 & 15 & 11
      # Note: in all experiments (except for computation time experiment), we use this version
      range_pred_st_array = range_iinfo_dict['izone_data'][:, [s_index, t_index]]

      check_dt :float= 0.05

      sample_s_from = parent_node[self._node_key_s]
      sample_s_to = np.max(child_nodes[:, self._node_key_s])
      sample_t_from = parent_node[self._node_key_t]
      sample_t_to = np.max(child_nodes[:, self._node_key_t])
      ref_sample_num = max(2, round((sample_s_to - sample_s_from) / self.path_s_interval))
      ref_sample_num = max(ref_sample_num, round((sample_t_to - sample_t_from) / check_dt))

      inquiry_s_array = np.linspace(sample_s_from, sample_s_to, ref_sample_num)
      check_stva_array = self._interpolate_nodes_given_s(
        parent_node, child_nodes, inquiry_s=inquiry_s_array)

      for ii, child_edge_stva in enumerate(check_stva_array):
        st_dists = np.array([range_pred_st_array - node_stva[[0, 1]] for node_stva in child_edge_stva])

        st_conditions = np.zeros_like(st_dists)
        st_conditions[:, :, 0] = self.path_protect_s
        st_conditions[:, :, 1] = self.yield_safe_time_gap

        stop_locs = child_edge_stva[:, 2] < self.stop_v_condition
        st_conditions[stop_locs, :, 1] = 1e+3 # those stop nodes will avoid collisions at whole future time

        s_collisions = np.logical_and(
          -st_conditions[:, :, 0] <= st_dists[:, :, 0],
          st_dists[:, :, 0] <= st_conditions[:, :, 0]
        )
        t_collisions = np.logical_and(
          -self.safe_time_gap < st_dists[:, :, 1],
          st_dists[:, :, 1] < st_conditions[:, :, 1]
        )

        collisions = np.logical_and(s_collisions, t_collisions)
        if np.sum(collisions) > 1e-3:
          valid_costs[ii, 0] = 0.0 # set 0 if collision

    return valid_costs
