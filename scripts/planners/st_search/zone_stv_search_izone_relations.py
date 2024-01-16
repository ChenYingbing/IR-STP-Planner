from typing import Dict, List, Tuple, Union
import numpy as np
import math
import copy
import abc

from planners.st_search.config import MAP_MODE2WEIGHTS
from planners.st_search.zone_stv_search import ZoneStvGraphSearch
from planners.interaction_space import InteractionFormat, InteractionSpace
from planners.st_search.sti_node import StiNode
from type_utils.agent import EgoAgent

from files.react_t2acc_table import assert_dict_name_is_legal, get_reaction_acc_values

TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE

class ZoneStvGraphSearchIZoneRelations(ZoneStvGraphSearch):
  '''
  Speed search based on the S-t-v graph, with interaction zone 
  modelling and priority determination relying on longitudinal responses along prediction results
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
                     algo_vars: Tuple = [0.0, 0.0, 0.0],
                     prediction_num: int = 1,
                     reaction_conditions: Dict = None,
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
    :param prediction_num: variable for algorithm
    :param reaction_conditions: reaction conditions for agents
    '''
    tmode = reaction_conditions['traj_mode']
    judge_init_relation = True
    if ('#' in tmode):
      tmode = tmode[:-1]
      judge_init_relation = False # disable judge init relation.
    self.__ignore_influ_cons = reaction_conditions['ignore_influ_cons']

    mode_st_coefficents = reaction_conditions['st_coefficents']
    if mode_st_coefficents <= 0:
      super().__init__(
        ego_agent, ispace, s_samples, xyyawcur_samples, start_sva, search_horizon_T, 
        planning_dt, prediction_dt, s_end_need_stop, friction_coef,
        path_s_interval, 
        enable_judge_init_relation=judge_init_relation,
        enable_debug_rviz=enable_debug_rviz)
    else:
      super().__init__(
        ego_agent, ispace, s_samples, xyyawcur_samples, start_sva, search_horizon_T, 
        planning_dt, prediction_dt, s_end_need_stop, friction_coef,
        path_s_interval, svaj_cost_weights=MAP_MODE2WEIGHTS[mode_st_coefficents],
        enable_judge_init_relation=judge_init_relation,
        enable_debug_rviz=enable_debug_rviz)
  
    # Algorithm variable
    self.__ireact_gap_cond_t = algo_vars[0]
    self.__ireact_delay_s = algo_vars[1]
    self.__influ_react_min_acc = algo_vars[2]
    # self.__ireact_coef = algo_vars[?] / float(prediction_num)
    
    self.__reaction_cond_acc = reaction_conditions['acc_const_value']
    self.__cond_low_speed = 0.2

    valid_modes = ['cvm', 'pred', 'irule']

    assert tmode in valid_modes,\
      "fatal value overtake_giveway_mode, {} / {}".format(tmode, valid_modes)
    self.__using_cvm = (tmode == 'cvm')
    self.__using_pred = (tmode == 'pred')
    self.__using_irule = (tmode == 'irule')

    self.__s_index = self.zone_part_data_len + InteractionFormat.iformat_index('av_s')
    self.__pred_s_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_s')
    self.__v0_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_v0')
    self.__t_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_t')
    self.__acc_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_acc')
    self.__iangle_index = self.zone_part_data_len + InteractionFormat.iformat_index('iangle')

    self.__agent_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_idx')
    self.__agent_traj_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_traj_idx')
    self.__tsv_indexs = StiNode.tsv_record_indexs_tiled()
    self.__tsv_t_indexs, self.__tsv_s_indexs, self.__tsv_v_indexs = StiNode.tsv_relative_idexes()

    self.__edge_check_func = None
    if self.__using_cvm:
      self.__edge_check_func = self.__edge_is_valid_cvm
    elif self.__using_pred:
      self.__edge_check_func = self.__edge_is_valid_pred
    elif self.__using_irule:
      self.__edge_check_func = self.__edge_is_valid_irule

  def _edge_is_valid(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the parent node's multiple child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    return self.__edge_check_func(range_iinfo_dict, parent_node, child_nodes)

  def __process_influ_constraints(self, locs_influ: np.ndarray, 
      check_stva_array: np.ndarray, child_izone_array: np.ndarray, valids_locs: np.ndarray):

    if np.sum(locs_influ) > 0:
      if self.__ignore_influ_cons:
        valids_locs[locs_influ] = True # set all true
      else:
        plan_t_array = check_stva_array[locs_influ, 1]
        pred_v0_array = child_izone_array[locs_influ, self.__v0_index]
        pred_s_array = child_izone_array[locs_influ, self.__pred_s_index]

        if (math.fabs(self.__influ_react_min_acc) >= 1e-1):
          # using cam
          ddd = np.square(pred_v0_array) + 2.0*self.__influ_react_min_acc*pred_s_array
          ddd_valids = ddd >= 0.0 # give dcc, vt >= 0.0
          iacc_arrive_t = np.ones_like(pred_v0_array) * 1e+3 # ddd_invalids are inf
          iacc_arrive_t[ddd_valids] = (-pred_v0_array[ddd_valids] + np.sqrt(ddd[ddd_valids])) / self.__influ_react_min_acc

          valids_locs[locs_influ] =\
            (iacc_arrive_t >= (plan_t_array + self.yield_safe_time_gap))
        else:
          # using cvm
          pred_v0_array[pred_v0_array < self._cvm_protect_min_v] = self._cvm_protect_min_v
          pred_cvm_t = pred_s_array / pred_v0_array

          valids_locs[locs_influ] =\
            (pred_cvm_t >= (plan_t_array + self.yield_safe_time_gap))

  def __process_relation_conflicts(self, range_iinfo_dict: Dict, get_relations: np.ndarray, valids_locs: np.ndarray) -> None:
    # set invalids when some edges have conflicted relations (e.g., have preempt and yield at meantime)

    for child_i, relation in enumerate(get_relations):
      # relation = [1, zone_and_interp_num], records relation_location_at_node x zone_and_interp_num
      #
      # cache shape = [zone_num, zone's multiple relation determination results]
      #   relation[ipoint_locs] returns [relation value] for each relation (zone)
      # 
      # > 0.3: aims to remove == 0.0, 0.25 relation (0.25 is influ relation)
      cache = [[v for v in relation[ipoint_locs] if math.fabs(v) > 0.3] \
        for relation_i, ipoint_locs in range_iinfo_dict['relation2ipoint_locs'].items()]
      # extract unqiue relations, e.g. [-1.0], [1.0], [-1.0, 1.0]
      # where [-1.0, 1.0] means have preempt and yield at meantime.

      # child_nodes.shape = (child_num, node_state_num)
      # valids_locs.shape = (child_num, zone_and_interp_num)
      # print(cache)
      # example = (4, 21) (4, 13) 2,
      #         = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
      # two interaction zones with [8, 5] overlaps with AV's planned trajectory,
      #   and their relations are [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0] respectively

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

  def __edge_is_valid_cvm(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the parent node's multiple child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0 # default with all edge valids

    if range_iinfo_dict['has_interaction'] and (child_nodes.shape[0] > 0):
      # prepare data
      check_izone_array = range_iinfo_dict['izone_data']
      if check_izone_array.shape[0] == 0:
        return valid_costs # no interactions, return all valids

      # relation calculations
      # check_izone_array with shape = (zone_and_interp_num, zone_info_dim)
      # check_stva_array with shape = (child_num, zone_and_interp_num, 4)
      # child_nodes with shape (child_num, node_state_num)
      check_stva_array = self._interpolate_nodes_given_s(
        parent_node, child_nodes, inquiry_s=check_izone_array[:, self.__s_index])

      # child_izone_array with shape = (child_num, zone_and_interp_num, zone_info_dim)
      child_izone_array =\
        np.repeat(np.array([check_izone_array.tolist()]), child_nodes.shape[0], axis=0)

      # child_relations with shape = (child_num, zone_and_interp_num)
      child_relations = child_nodes[:, range_iinfo_dict['relation_indexes']]
      locs_notdeter = np.abs(child_relations - StiNode.relation_not_determined()) < 1e-3

      # init relations with original values
      get_relations = child_relations.copy()
      valids_locs = np.zeros_like(child_relations) # default all are invalids

      ################################################################
      # not determined locations
      if np.sum(locs_notdeter) > 0:
        plan_t_array = check_stva_array[locs_notdeter, 1]
        pred_v0_array = child_izone_array[locs_notdeter, self.__v0_index]
        pred_s_array = child_izone_array[locs_notdeter, self.__pred_s_index]

        pred_v0_array[pred_v0_array < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / pred_v0_array

        preempt_locs = plan_t_array <= (pred_cvm_t - self.safe_time_gap)
        yield_locs = plan_t_array >= (pred_cvm_t + self.yield_safe_time_gap)

        # update relationship
        #   preempt_locs with relation_preempt
        #   yield_locs with relation_yield
        #   others with original values
        get_relations[locs_notdeter] =\
          preempt_locs * StiNode.relation_preempt() +\
          yield_locs * StiNode.relation_yield()

        self._update_relations(child_nodes, range_iinfo_dict, get_relations)
        valids_locs[locs_notdeter] =\
          np.logical_or(preempt_locs, yield_locs) * 1.0 # set valids

      ################################################################
      # influ locations
      locs_influ = np.abs(get_relations - StiNode.relation_influ()) < 1e-3

      self.__process_influ_constraints(
        locs_influ, check_stva_array, child_izone_array, valids_locs)

      ################################################################
      # preempt locations
      locs_preempt = np.abs(get_relations - StiNode.relation_preempt()) < 1e-3

      if np.sum(locs_preempt) > 0:
        plan_t_array = check_stva_array[locs_preempt, 1]
        plan_v_array = check_stva_array[locs_preempt, 2]
        pred_v0_array = child_izone_array[locs_preempt, self.__v0_index]
        pred_s_array = child_izone_array[locs_preempt, self.__pred_s_index]

        pred_v0_array[pred_v0_array < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / pred_v0_array

        # preempt means that agent's t exist > plan_t, if av plan to stop, it would be invalid
        not2stop_array = plan_v_array > self.stop_v_condition
        valids_locs[locs_preempt] = np.logical_and(
          plan_t_array <= (pred_cvm_t - self.safe_time_gap), 
          not2stop_array)

      ################################################################
      # yield locations
      locs_yield = np.abs(get_relations - StiNode.relation_yield()) < 1e-3

      if np.sum(locs_yield) > 0:
        plan_t_array = check_stva_array[locs_yield, 1]
        pred_v0_array = child_izone_array[locs_yield, self.__v0_index]
        pred_s_array = child_izone_array[locs_yield, self.__pred_s_index]

        pred_v0_array[pred_v0_array < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / pred_v0_array

        valids_locs[locs_yield] = plan_t_array >= (pred_cvm_t + self.yield_safe_time_gap)
        # set valids when is2stop and collisions at any future time stamp
        # because: any stop is valid as it already arrives after other agents

      ################################################################
      # set invalids when some edges have conflicted relations (e.g., have preempt and yield at meantime)
      self.__process_relation_conflicts(range_iinfo_dict, get_relations, valids_locs)

      # set valid & costs
      illegal_child_locs = np.sum(valids_locs, axis=1) < valids_locs.shape[1]
      valid_costs[illegal_child_locs, 0] = 0.0 # set invalid

    return valid_costs

  def __edge_is_valid_pred(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the parent node's multiple child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0 # default with all edge valids

    if range_iinfo_dict['has_interaction'] and (child_nodes.shape[0] > 0):
      # prepare data
      check_izone_array = range_iinfo_dict['izone_data']
      if check_izone_array.shape[0] == 0:
        return valid_costs # no interactions, return all valids

      # relation calculations
      # check_izone_array with shape = (zone_and_interp_num, zone_info_dim)
      # check_stva_array with shape = (child_num, zone_and_interp_num, 4)
      # child_nodes with shape (child_num, node_state_num)
      check_stva_array = self._interpolate_nodes_given_s(
        parent_node, child_nodes, inquiry_s=check_izone_array[:, self.__s_index])

      # child_izone_array with shape = (child_num, zone_and_interp_num, zone_info_dim)
      child_izone_array =\
        np.repeat(np.array([check_izone_array.tolist()]), child_nodes.shape[0], axis=0)

      # child_relations with shape = (child_num, zone_and_interp_num)
      child_relations = child_nodes[:, range_iinfo_dict['relation_indexes']]
      locs_notdeter = np.abs(child_relations - StiNode.relation_not_determined()) < 1e-3

      # init relations with original values
      get_relations = child_relations.copy()
      valids_locs = np.zeros_like(child_relations) # default all are invalids

      ################################################################
      # not determined locations
      if np.sum(locs_notdeter) > 0:
        plan_t_array = check_stva_array[locs_notdeter, 1]
        pred_t_array = child_izone_array[locs_notdeter, self.__t_index]

        preempt_locs = plan_t_array <= (pred_t_array - self.safe_time_gap)
        yield_locs = plan_t_array >= (pred_t_array + self.yield_safe_time_gap)

        # update relationship
        #   preempt_locs with relation_preempt
        #   yield_locs with relation_yield
        #   others with original values
        get_relations[locs_notdeter] =\
          preempt_locs * StiNode.relation_preempt() + yield_locs * StiNode.relation_yield()

        self._update_relations(child_nodes, range_iinfo_dict, get_relations)
        valids_locs[locs_notdeter] =\
          np.logical_or(preempt_locs, yield_locs) * 1.0 # set valids

      ################################################################
      # influ locations
      locs_influ = np.abs(get_relations - StiNode.relation_influ()) < 1e-3

      self.__process_influ_constraints(
        locs_influ, check_stva_array, child_izone_array, valids_locs)

      ################################################################
      # preempt locations
      locs_preempt = np.abs(get_relations - StiNode.relation_preempt()) < 1e-3

      if np.sum(locs_preempt) > 0:
        plan_t_array = check_stva_array[locs_preempt, 1]
        plan_v_array = check_stva_array[locs_preempt, 2]
        pred_t_array = child_izone_array[locs_preempt, self.__t_index]

        # preempt means that agent's t exist > plan_t, if av plan to stop, it would be invalid
        not2stop_array = plan_v_array > self.stop_v_condition
        valids_locs[locs_preempt] = np.logical_and(
          plan_t_array <= (pred_t_array - self.safe_time_gap), 
          not2stop_array)

      ################################################################
      # yield locations
      locs_yield = np.abs(get_relations - StiNode.relation_yield()) < 1e-3

      if np.sum(locs_yield) > 0:
        plan_t_array = check_stva_array[locs_yield, 1]
        pred_t_array = child_izone_array[locs_yield, self.__t_index]

        valids_locs[locs_yield] =\
          (plan_t_array >= (pred_t_array + self.yield_safe_time_gap))
  
        # set valids when is2stop and collisions at any future time stamp
        # because: any stop is valid as it already arrives after other agents

      ################################################################
      # set invalids when some edges have conflicted relations (e.g., have preempt and yield at meantime)
      self.__process_relation_conflicts(range_iinfo_dict, get_relations, valids_locs)

      # set valid & costs
      illegal_child_locs = np.sum(valids_locs, axis=1) < valids_locs.shape[1]
      valid_costs[illegal_child_locs, 0] = 0.0 # set invalid

    return valid_costs

  def __edge_is_valid_irule(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the parent node's multiple child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0 # default with all edge valids

    plan_v0 = self.start_sva[1]
    if range_iinfo_dict['has_interaction'] and (child_nodes.shape[0] > 0):
      # prepare data
      check_izone_array = range_iinfo_dict['izone_data']
      # area_protect_s_extended = range_iinfo_dict['area_protect_s_extended']
      # print(check_izone_array[:, [0, 1, 3, 4, 6, 7]]) agent/agent_traj s/t values overlapped with path point of AV
      if check_izone_array.shape[0] == 0:
        return valid_costs # no interactions, return all valids

      # relation calculations
      # check_izone_array with shape = (zone_and_interp_num, zone_info_dim)
      # check_stva_array with shape = (child_num, zone_and_interp_num, 4)
      # record_tsv_array with shape = (child_num, zone_and_interp_num, record_num x 3)
      # child_nodes with shape (child_num, node_state_num)
      check_stva_array = self._interpolate_nodes_given_s(
        parent_node, child_nodes, inquiry_s=check_izone_array[:, self.__s_index])
      # record_tsv_array = self.__get_tsv_record(
      #   child_nodes=child_nodes, inquiry_s=check_izone_array[:, self.__s_index])

      # child_izone_array with shape = (child_num, zone_and_interp_num, zone_info_dim)
      child_izone_array =\
        np.repeat(np.array([check_izone_array.tolist()]), child_nodes.shape[0], axis=0)

      # child_relations with shape = (child_num, zone_and_interp_num)
      child_relations = child_nodes[:, range_iinfo_dict['relation_indexes']]
      locs_notdeter = np.abs(child_relations - StiNode.relation_not_determined()) < 1e-3
      # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
      # # example = (4, 13) (4, 13, 11) 13
      # #         = (13, 11)
      # print(child_relations.shape, child_izone_array.shape, len(range_iinfo_dict['relation_indexes']))
      # print(check_izone_array.shape)

      # init relations with original values
      get_relations = child_relations.copy()
      valids_locs = np.zeros_like(child_relations) # default all are invalids
      costs_locs = np.zeros_like(child_relations)
      notdter_set_inlu_locs = valids_locs < -1e+3 # init with all false

      ################################################################
      # not determined locations
      if np.sum(locs_notdeter) > 0:
        plan_t_array = check_stva_array[locs_notdeter, 1]
        plan_v_array = check_stva_array[locs_notdeter, 2]
        pred_v0_array = child_izone_array[locs_notdeter, self.__v0_index]
        pred_s_array = child_izone_array[locs_notdeter, self.__pred_s_index]

        # protect_v0 = pred_v0_array.copy()
        # protect_v0[protect_v0 < self._cvm_protect_min_v] = self._cvm_protect_min_v
        # pred_cvm_t = pred_s_array / protect_v0
        
        pred_t_array = child_izone_array[locs_notdeter, self.__t_index]
        # predcvm_t_array = pred_t_array.copy()
        # replace_locs = pred_v0_array < self.__cond_low_speed
        # predcvm_t_array[replace_locs] = pred_cvm_t[replace_locs]

        ddd = np.square(pred_v0_array) + 2.0*self.__reaction_cond_acc*pred_s_array
        ddd_valids = ddd >= 0.0 # give dcc, vt >= 0.0
        iacc_arrive_t = np.ones_like(pred_v0_array) * 1e+3 # ddd_invalids are inf
        iacc_arrive_t[ddd_valids] = (-pred_v0_array[ddd_valids] + np.sqrt(ddd[ddd_valids])) / self.__reaction_cond_acc

        influ_locs = np.logical_and(
          (plan_t_array + self.__ireact_delay_s / (1e-3 + plan_v_array)) <= (pred_t_array - self.__ireact_gap_cond_t),
          iacc_arrive_t >= (plan_t_array + self.yield_safe_time_gap)
        )

        preempt_locs = plan_t_array <= (pred_t_array - self.safe_time_gap)
        yield_locs = plan_t_array >= (pred_t_array + self.yield_safe_time_gap)
        preempt_locs[influ_locs] = False
        yield_locs[influ_locs] = False

        valid_array1 = np.logical_or(yield_locs, influ_locs)
        notdter_set_inlu_locs[locs_notdeter] = influ_locs

        valids_locs[locs_notdeter] =\
          np.logical_or(preempt_locs, valid_array1) * 1.0 # set valids

        # update relationship
        #   preempt_locs with relation_preempt
        #   yield_locs with relation_yield
        #   others with original values
        get_relations[locs_notdeter] =\
          preempt_locs * StiNode.relation_preempt() +\
          influ_locs * StiNode.relation_influ() +\
          yield_locs * StiNode.relation_yield()
        self._update_relations(child_nodes, range_iinfo_dict, get_relations)

      ################################################################
      # influ locations
      locs_influ = np.abs(get_relations - StiNode.relation_influ()) < 1e-3

      self.__process_influ_constraints(
        locs_influ, check_stva_array, child_izone_array, valids_locs)

      ################################################################
      # preempt locations
      locs_preempt = np.abs(get_relations - StiNode.relation_preempt()) < 1e-3

      if np.sum(locs_preempt) > 0:
        plan_t_array = check_stva_array[locs_preempt, 1]
        plan_v_array = check_stva_array[locs_preempt, 2]
        pred_t_array = child_izone_array[locs_preempt, self.__t_index]

        # preempt means that agent's t exist > plan_t, if av plan to stop, it would be invalid
        not2stop_array = plan_v_array > self.stop_v_condition

        valids_locs[locs_preempt] =\
          np.logical_and(plan_t_array <= (pred_t_array - self.safe_time_gap), 
          not2stop_array)
        # print("locs_preempt", valids_locs[locs_preempt])

      ################################################################
      # yield locations
      locs_yield = np.abs(get_relations - StiNode.relation_yield()) < 1e-3

      if np.sum(locs_yield) > 0:
        plan_t_array = check_stva_array[locs_yield, 1]
        pred_t_array = child_izone_array[locs_yield, self.__t_index]

        valids_locs[locs_yield] = plan_t_array >= (pred_t_array + self.yield_safe_time_gap)
        # set valids when is2stop and collisions at any future time stamp
        # because: any stop is valid as it already arrives after other agents

      ################################################################
      # set invalids when some edges have conflicted relations (e.g., have preempt and yield at meantime)
      self.__process_relation_conflicts(range_iinfo_dict, get_relations, valids_locs)

      # set valid & costs
      illegal_child_locs = np.sum(valids_locs, axis=1) < valids_locs.shape[1]
      valid_costs[illegal_child_locs, 0] = 0.0 # set invalid
      valid_costs[:, 1] = np.sum(costs_locs, axis=1) # fill costs

    return valid_costs
