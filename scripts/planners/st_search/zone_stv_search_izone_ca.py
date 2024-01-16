from typing import Dict, List, Tuple, Union
import numpy as np
import math
import copy
import abc

from planners.st_search.zone_stv_search import ZoneStvGraphSearch
from planners.interaction_space import InteractionFormat, InteractionSpace
from planners.st_search.sti_node import StiNode
from type_utils.agent import EgoAgent

class ZoneStvGraphSearchIZoneCollisionAvoidance(ZoneStvGraphSearch):
  '''
  Speed search based on the S-t-v graph, with interaction zone 
  modelling and priority determination relying on collision checking with prediction results
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
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the edge's child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    # @note this code is identical to 'pred' as relation judgment in zone_stv_search_izone_relations.py
    # (we write this version first in developing, then migrated to zone_stv_search_izone_relations.py)
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0 # default with all edge valids

    if range_iinfo_dict['has_interaction'] and (child_nodes.shape[0] > 0):
      # prepare data
      check_izone_array = range_iinfo_dict['izone_data']
      if check_izone_array.shape[0] == 0:
        return valid_costs # no interactions, return all valids

      s_index = self.zone_part_data_len + InteractionFormat.iformat_index('av_s')
      t_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_t')

      # relation calculations
      # check_izone_array with shape = (edge_check_point_num, zone_info_num)
      # check_stva_array with shape = (child_num, edge_check_point_num, 4)
      # child_nodes with shape (child_num, node_state_num)
      check_stva_array = self._interpolate_nodes_given_s(
        parent_node, child_nodes, inquiry_s=check_izone_array[:, s_index])

      # child_izone_array with shape = (child_num, edge_check_point_num, zone_info_num)
      child_izone_array = np.repeat(np.array([check_izone_array.tolist()]), child_nodes.shape[0], axis=0)

      # child_relations with shape = (child_num, edge_check_point_num)
      child_relations = child_nodes[:, range_iinfo_dict['relation_indexes']]

      # locs_notdeter = np.abs(child_relations) < 1e-3
      locs_notdeter = np.abs(child_relations - StiNode.relation_not_determined()) < 1e-3

      # init relations with original values
      get_relations = child_relations.copy()
      valids_locs = np.zeros_like(child_relations)

      ################################################################
      # not determined locations
      if np.sum(locs_notdeter) > 0:
        plan_t_array = check_stva_array[locs_notdeter, 1]
        plan_v_array = check_stva_array[locs_notdeter, 2]
        pred_t_array = child_izone_array[locs_notdeter, t_index]

        ovtk_locs = plan_t_array <= (pred_t_array - self.safe_time_gap)
        yield_locs = plan_t_array >= (pred_t_array + self.yield_safe_time_gap)

        valids_locs[locs_notdeter] = np.logical_or(ovtk_locs, yield_locs) * 1.0

        # update relationship
        #   ovtk_locs with relation_preempt
        #   yield_locs with relation_yield
        #   others with 0.0, not determined
        get_relations[locs_notdeter] =\
          ovtk_locs * StiNode.relation_preempt() + yield_locs * StiNode.relation_yield()

        self._update_relations(child_nodes, range_iinfo_dict, get_relations)

      ################################################################
      # preempt locations
      locs_preempt = np.abs(get_relations - StiNode.relation_preempt()) < 1e-3

      if np.sum(locs_preempt) > 0:
        plan_t_array = check_stva_array[locs_preempt, 1]
        plan_v_array = check_stva_array[locs_preempt, 2]
        pred_t_array = child_izone_array[locs_preempt, t_index]
        
        # preempt means that agent's t exist > plan_t, if av plan to stop, it would be invalid
        not2stop_array = plan_v_array > self.stop_v_condition
        valids_locs[locs_preempt] =\
          np.logical_and(
            plan_t_array <= (pred_t_array - self.safe_time_gap),
            not2stop_array)

      ################################################################
      # yield locations
      locs_yield = np.abs(get_relations - StiNode.relation_yield()) < 1e-3

      if np.sum(locs_yield) > 0:
        plan_t_array = check_stva_array[locs_yield, 1]
        pred_t_array = child_izone_array[locs_yield, t_index] 

        valids_locs[locs_yield] =\
          (plan_t_array >= (pred_t_array + self.yield_safe_time_gap))

        # set valids when is stop and collisions at any future time stamp
        # because: any stop is valid as it already arrives after other agents

      ################################################################
      # set invalids when some edges have ovtk and giveway at meantime.

      for child_i, relation in enumerate(get_relations):
        # > 0.3: aims to remove == 0.0, 0.25 relation
        cache = [[v for v in relation[seq_locs] if math.fabs(v) > 0.3] \
          for relation_i, seq_locs in range_iinfo_dict['relation2ipoint_locs'].items()]
        # extract unqiue relations, e.g. [-1.0], [1.0], [-1.0, 1.0]
        # where [-1.0, 1.0] means have ovtk and giveway at meantime.
        
        # check ipoints corresponding to each relation_index has unique preempt/yeild relation 
        # influ is ignored when couting the relation number
        cache_unique_len = []
        for cc in cache:
          unique_cc = np.unique(cc)
          unique_cc_list = unique_cc.tolist()
          len_unique_values :int= unique_cc.shape[0]
          if (StiNode.relation_influ() in unique_cc_list) and (StiNode.relation_preempt() in unique_cc_list):
            len_unique_values -= 1
          cache_unique_len.append(len_unique_values)

        max_unique_num = max(cache_unique_len)
        if max_unique_num >= 2:
          valids_locs[child_i, :] = 0.0 # set all invalid along of this edge check point

      # set valid & costs
      illegal_child_locs = np.sum(valids_locs, axis=1) < valids_locs.shape[1]
      valid_costs[illegal_child_locs, 0] = 0.0 # set invalid

    return valid_costs
