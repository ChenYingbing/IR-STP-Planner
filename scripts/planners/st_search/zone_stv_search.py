from typing import Dict, List, Tuple, Union
import numpy as np
import math
import copy
import abc
from multiprocessing.pool import ThreadPool

from planners.st_search.sti_node import StiNode
from planners.st_search.metric_svaj import SVAJCostMetric
from utils.angle_operation import get_mean_yaw_value, get_normalized_angle
from type_utils.agent import EgoAgent

from planners.interaction_space import InteractionFormat, InteractionSpace
from pymath.constraints.speed_cur_cons import SpeedCurvatureCons
from utils.colored_print import ColorPrinter

M_PI_2 = 1.5707963267948966
TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE

MAX_SPEED_LIMIT = 30.0

class ZoneStvGraphSearch:
  '''
  Speed search based on the S-t-v graph 
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
                     svaj_cost_weights: Tuple = [-0.0, 5.0, 0.5, 0.8],
                     ego_front_length_inflation: float = 0.25,
                     enable_judge_init_relation = False,
                     one_predtraj_multi_izone = True,
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
    :param svaj_cost_weights: weights to calculate sum_dist, velocity cost to speed limit, acc, jerk costs in A*
           @note: acc cost = acc**2 * dt; jerk cost = jerk**2 * dt;
    :param enable_judge_init_relation: judge relation according to agent's initial relative locations
    :param one_predtraj_multi_izone: if set false, one prediction trajectory only has one interaction zone
    '''
    self.ego_agent = ego_agent
    self.ispace = ispace
    self.search_horizon_T = search_horizon_T
    self.planning_dt = planning_dt
    self.prediction_dt = prediction_dt

    self.path_s_interval = path_s_interval
    self.path_s_interval_2 = path_s_interval * 0.5
    
    self.enable_judge_init_relation = enable_judge_init_relation
    self.one_predtraj_multi_izone = one_predtraj_multi_izone
    self.enable_debug_rviz = enable_debug_rviz

    # parameters to adjust search efficiency
    self.max_v_limit = self.get_max_v_limit()
    self.search_max_v_sample_num = 20  # maximum forward sampling number at each forward expansion
    self.search_jerk_limit = 8.0       # jerk limit to abandon invalid edges
    self.safe_time_gap = 0.5           # preempt protection time
    self.yield_safe_time_gap = 0.5     # yield protection time

    self.goal_cond_reso = 0.5
    self.goal_cond_tstamp = self.search_horizon_T - 0.5
    self.stop_v_condition = 0.1
    self.result_traj_max_num :int= 5

    s_v_limits = SpeedCurvatureCons.get_curvature_speed_limits(
      curvatures=xyyawcur_samples[:, 3], u=friction_coef,
      min_speed_limit=5.0)
    if s_end_need_stop == True:
      s_v_limits[-1] = 1e-3

    ## variables
    self._cvm_protect_min_v = 0.1
    self.start_sva = start_sva

    # records, such as nodes' cost at each layer
    self.search_maximum_layer: int = 0
    self.search_max_speed: float = 0.0
    self.layer_node_cost_record :Dict[int, List[Tuple]]= {}    # map from s sample index > list of [node_index, cost]
    self.node_records :Dict[int, np.ndarray]= {}               # map from node index to node state array
    self.edge_pair_records :Dict[Tuple(int, int), float]= {}   # map valid edge's (parent_node_index, child_node_index) to their costs

    self.goal_records :Dict[float, List] = {} # map from goal time value to list of goal nodes

    # s-t-v relevant things
    self.s_index_reso :float= 0.2

    self.search_t_reso = 0.2
    self.search_t_reso_2 = self.search_t_reso * 0.5
    self.t_axis_grid_num :int= math.ceil(self.search_horizon_T / self.search_t_reso)

    self.search_ref_v_reso = 0.3
    self.search_ref_v_reso_2 = self.search_ref_v_reso * 0.5
    self.v_axis_grid_num :int= math.ceil(self.max_v_limit / self.search_ref_v_reso)

    self.search_acc_bounds = [-4.0, 3.0]
    self.tv_space_grid_num: int= self.t_axis_grid_num * self.v_axis_grid_num

    # init things
    self.ego_length = self.ego_agent.info['length']
    self.ego_half_length = self.ego_length * 0.5
    self.ego_width = self.ego_agent.info['width']
    self.ego_half_width = self.ego_width * 0.5

    self.path_protect_s = (path_s_interval * 2.1) # protection distance
    self.ego_front_length_inflation = ego_front_length_inflation

    self.izone_area_cond_s :float= 5.0
    # self.izone_area_cond_t :float= 1.5
    self.inv_line_overlap_cond_radian = 165.0 * TO_RADIAN

    self.ori_s_samples = s_samples
    self.__init_interaction_info(s_samples, xyyawcur_samples)
    self.s_samples, self.xyyaw_samples, self.v_limits =\
      self.__downsample_s_samples(
        start_sva, s_samples, xyyawcur_samples, s_v_limits, 
        max_v_limit=self.max_v_limit)
    self.s_tv_bounds = self.__init_s_sample_tv_bounds(start_sva=start_sva)

    self.s_sample_bias = self.s_samples[0]
    self.s_axis_grid_num: int= self.s_samples.shape[0]
    self.s_maximum_layer_idx: int= self.s_samples.shape[0] - 1

    self.__tsv_indexs = np.array(StiNode.tsv_record_indexs(), dtype=int)

    # cost items
    self.__cost_metric = SVAJCostMetric(
      start_s=self.s_sample_bias, goal_s=self.s_samples[-1], 
      speed_limits=self.v_limits, svaj_weights=np.array(svaj_cost_weights))

    # rviz items
    self.__enable_plot_plan_results = True

    self._rviz_dict = {}
    self.__rviz_plan_results = []

    # debug

  def reinit(self) -> None:
    '''
    Reinit the class to prepare for next search
    '''
    self.search_maximum_layer = 0
    self.search_max_speed = 0.0

    self.layer_node_cost_record = {}
    self.node_records = {}
    self.edge_pair_records = {}

    self.goal_records = {}

    self._reinit()

    self._rviz_dict = {}
    self.__rviz_plan_results = []

  @abc.abstractmethod
  def _reinit(self) -> None:
    '''
    Reinit things function, which can be inherited by child class
    '''
    pass
  
  @staticmethod
  def get_max_v_limit() -> float:
    '''
    Return maximum speed value in searching
    '''
    return MAX_SPEED_LIMIT

  def get_search_acc_bounds(self) -> List:
    '''
    Return acc bounds [acc_lower_bound, acc_upper_bound]
    '''
    return self.search_acc_bounds

  def set_search_acc_bounds(self, acc_bounds: Tuple[float, float]) -> None:
    '''
    Set search acc lower bound and upper bound
    :param acc_bounds: [acc_lower_bound, acc_upper_bound], if bound value is
           nan, skip to set the corresponding value.
    '''
    if isinstance(acc_bounds[0], float):
      self.search_acc_bounds[0] = acc_bounds[0]
    if isinstance(acc_bounds[1], float):
      self.search_acc_bounds[1] = acc_bounds[1]

  def _enable_plot_plan_results(self, enable_flag: bool) -> None:
    '''
    Enable to plot plan results or not.
    '''
    self.__enable_plot_plan_results = enable_flag

  def __downsample_s_samples(self, start_sva: Tuple[float, float, float],
                                   s_samples: np.ndarray, 
                                   xyyaw_samples: np.ndarray,
                                   s_v_limits: np.ndarray, 
                                   max_v_limit: float) -> Tuple:
    '''
    Downsampling s_samples and their speed limits
    :param start_sva: initial s,v,a values of AV
    :param s_samples: s array to sample v limits
    :param xyyaw_samples: xyyaw values of s samples
    :param s_v_limits: v limits at each s samples
    :param max_v_limit: maximum v limit for the subsampled s array
    :return: downsampled s_samples, xyyaw_samples, and their speed limits
    '''
    assert s_samples.shape[0] == s_v_limits.shape[0], "Size unequal."

    # preparations
    # s = 0.5 * (0**2 - v0**2) / dcc
    not_add_after_min_stop_s = True
    min_stop_s = 0.5 * (start_sva[1] ** 2) / (-self.search_acc_bounds[0])

    limit_conv_distance = 8.0
    max_interval_vlimit = 3.0
    
    ref_sample_dt = 1.0
    ref_sample_dcc = -1.0
    ref_sample_dcc_dur_t = 2.0
    min_interval_s = 2.0
    # max(v_min, v0 + ref_dcc * dt) * dt 
    max_interval_s = max(min_interval_s,
      max(1.0, start_sva[1] + ref_sample_dcc * ref_sample_dcc_dur_t) * ref_sample_dt)
    max_interval_s = round(max_interval_s, 1)

    conv_N = round(limit_conv_distance / self.path_s_interval)
    extentd_num = round(conv_N * 0.5)
    mean_s_v_limits = np.convolve(
      np.array(
        [s_v_limits[0]]*extentd_num + s_v_limits.tolist() + [s_v_limits[-1]]*extentd_num
      ), np.ones(conv_N)/conv_N, mode='valid')
    mean_s_v_limits = mean_s_v_limits[:s_samples.shape[0]]
    assert mean_s_v_limits.shape[0] == s_samples.shape[0], "Size unequal"

    mean_s_v_limits[mean_s_v_limits > max_v_limit] = max_v_limit # limit max values

    # subsample s_samples
    len_1 = mean_s_v_limits.shape[0] - 1
    ii :int= 0
    get_s_vlimits = []
    get_xyyaw_samples = []
    for s_value, xyyaw, v_limit in zip(s_samples, xyyaw_samples, mean_s_v_limits):
      if len(get_s_vlimits) == 0:
        get_s_vlimits.append([s_value, v_limit])
      else:
        last_s_vlimit = get_s_vlimits[-1]
        ds = math.fabs(s_value - last_s_vlimit[0])
        dv_limit = math.fabs(v_limit - last_s_vlimit[1])

        enable_pop_one = (ds <= min_interval_s) and (len(get_s_vlimits) > 1)
        if math.fabs(s_value - min_stop_s) <= self.path_s_interval_2:
          # forcibly append min_stop_s nearby s sample
          if enable_pop_one:
            get_s_vlimits.pop()
            get_xyyaw_samples.pop()
          get_s_vlimits.append([s_value, v_limit])
          get_xyyaw_samples.append([xyyaw[0], xyyaw[1], xyyaw[2]])
        else:
          if (ds >= min_interval_s) and ((ds >= max_interval_s) or (dv_limit >= max_interval_vlimit)):
            # append when meets conditions
            get_s_vlimits.append([s_value, v_limit])
            get_xyyaw_samples.append([xyyaw[0], xyyaw[1], xyyaw[2]])
          elif (ii == len_1): # last one operations
            if enable_pop_one:
              get_s_vlimits.pop() # check if pop last one
              get_xyyaw_samples.pop()
            get_s_vlimits.append([s_value, v_limit])
            get_xyyaw_samples.append([xyyaw[0], xyyaw[1], xyyaw[2]])
      ii += 1

    get_s_vlimits = np.array(get_s_vlimits)
    get_xyyaw_samples = np.array(get_xyyaw_samples)

    # print("get get_s_samples", min_stop_s)
    # print(get_s_vlimits[:, 0])
    return get_s_vlimits[:, 0], get_xyyaw_samples, get_s_vlimits[:, 1]

  def __init_s_sample_tv_bounds(self, start_sva: Tuple[float, float, float]) -> np.ndarray:
    '''
    Init v bound values at each s samples
    :param start_sva: initial s,v,a values of AV
    :return: [[s, t_lower_bound, t_upper_bound, v_lower_bound, v_upper_bound], ...] np array
    '''
    if self.s_samples.shape[0] == 0:
      return np.array([])

    # initial with [v_lower_bound=v0, v_upper_bound=v0]
    cache_vbounds = [start_sva[1], start_sva[1]]
    cache_tbounds = [0.0, 0.0]

    s_tv_bounds = [[self.s_samples[0], cache_tbounds[0], cache_tbounds[1], cache_vbounds[0], cache_vbounds[1]]]
    max_t_bound = self.search_horizon_T + 4.0

    last_s :float= s_tv_bounds[-1][0]
    for s, vlimit in zip(self.s_samples[1:], self.v_limits[1:]):
      delta_s = s - last_s

      ## vt**2 = v0**2 + 2*a*s
      ## t = s / (0.5*v0 + 0.5*vt)
      # calculate v upper bound
      v0 = cache_vbounds[1]
      vt_u = self.max_v_limit
      if v0 < self.max_v_limit:
        max_vt = math.sqrt(
          v0**2 + 2.0 * self.search_acc_bounds[1] * delta_s)
        vt_u = min(max_vt, min(vlimit, self.max_v_limit))
      min_delta_t = delta_s / (0.5*v0 + 0.5*vt_u)

      # calculate v lower bound
      v0 = cache_vbounds[0]
      vt_l = 0.0
      if v0 > 0.0:
        min_vt = math.sqrt(
          max(0.0, v0**2 + 2.0 * self.search_acc_bounds[0] * delta_s)          
        )
        vt_l = min_vt
      max_delta_t = delta_s / max(1e-2, 0.5*v0 + 0.5*vt_l)
      
      # calculate t upper bound
      t0 = cache_tbounds[1]
      t_u = min(max_t_bound, t0+max_delta_t)

      # calculate t lower bound
      t0 = cache_tbounds[0]
      t_l = t0 + min_delta_t

      cache_vbounds = [vt_l, vt_u]
      cache_tbounds = [t_l, t_u]
      s_tv_bounds.append([s, t_l, t_u, vt_l, vt_u])

    assert len(s_tv_bounds) == self.s_samples.shape[0], "Fatal error, size unequal."
    return np.array(s_tv_bounds)

  def __init_interaction_info(self, s_samples: np.ndarray, xyyawcur_samples: np.ndarray) -> None:
    '''
    Init interaction information, data formats pls see details in the function 
    '''
    assert s_samples.shape[0] == xyyawcur_samples.shape[0], "Size unequal."
    # configs
    self.zone_s_reso = 0.2

    # get agent2traj2iinfos: map from agent_id > traj_id > interaction informations 
    agent2traj2iinfos: Dict[int, Dict[int, Dict]] = {}

    formmater = InteractionFormat(plan_horizon_T=self.search_horizon_T,
      plan_dt=self.planning_dt, prediction_dt=self.prediction_dt)

    # TODO(abing): reduce the interaction point numbers of each agent to increase the calculation efficiency.
    iinfo_list, av_rear_already_occupied_dict, av_front_already_occupied_dict = self.ispace.read_interactions(
      self.ego_length, self.ego_width, self.ego_front_length_inflation, xyyawcur_samples)
    for av_s, iinfo in zip(s_samples, iinfo_list):
      if iinfo['is_interacted']:
        # has potential conflicts with other traffic participants.
        path_s = max(av_s - (self.ego_length * 0.5), 0.0)
        formmater.add_iinfo2dict(path_s, iinfo['details'], agent2traj2iinfos)

    # init interaction zones & extract interaction data
    izone_data_list :List= []              # interaction zone data
    izone_key2node_relation_index :Dict[Tuple, int]= {}    # map from zone_key to relation index in node reprentations
    relation_index2izone_key :Dict[int, Tuple]= {} 
    izone_key2points :Dict[Tuple, List]= {}       # map from zone_key to list of point indexes
    # izone_key2points:
    #   map from= zone_key=(agent_index, trajectory_index, zone_index),
    #   to= list of interaction zone data's row index
    
    init_influ_indexes = []
    init_yield_indexes = []

    format_agent_idx :int= InteractionFormat.iformat_index('agent_idx')
    format_agent_trajidx :int= InteractionFormat.iformat_index('agent_traj_idx')
    format_av_sidx :int= InteractionFormat.iformat_index('av_s')
    format_agent_v0idx :int= InteractionFormat.iformat_index('agent_v0')
    format_agent_sidx :int= InteractionFormat.iformat_index('agent_s')
    format_agent_tidx :int= InteractionFormat.iformat_index('agent_t')
    format_iangle_idx :int= InteractionFormat.iformat_index('iangle')

    relation_index :int = 0
    for agent_idx, trajs_dict in agent2traj2iinfos.items():
      for traj_idx, dict_data in trajs_dict.items():
        full_iinfo_array =\
          np.array(agent2traj2iinfos[agent_idx][traj_idx]['s/id_trajid_v0_s_t_v_acc_iangle'])

        # get each interaction zone's initial point index: zone_from_points
        # if (agent_idx == 301660):
        #   print("inv_cond=", self.inv_line_overlap_cond_radian)
        #   print("full_iinfo_array s-s-t-ia", full_iinfo_array[:, [0,4,5,-1]])

        last_zone_is_inv_zone = False
        zone_from_points = [0]
        if self.one_predtraj_multi_izone == True:
          for ii in range(1, full_iinfo_array.shape[0]):
            dt0 = full_iinfo_array[ii-1]
            dt1 = full_iinfo_array[ii]

            delta_path_s = math.fabs(dt1[format_av_sidx] - dt0[format_av_sidx])
            # delta_pred_t = math.fabs(dt1[format_agent_tidx] - dt0[format_agent_tidx])
            # if (delta_path_s > self.izone_area_cond_s) and (delta_pred_t > self.izone_area_cond_t):
            if (delta_path_s > self.izone_area_cond_s):
              # start a new zone when
              #   cond1: path_s is not continuous
              #   cond2: agent arrive t is not continuous
              zone_from_points.append(ii)
              last_zone_is_inv_zone = False

            else:
              if last_zone_is_inv_zone:
                # a inv_zone cannot occupy too much distance, should be splited into two
                last_zone_first_point = full_iinfo_array[zone_from_points[-1]]
                if (math.fabs(dt1[format_iangle_idx]) >= self.inv_line_overlap_cond_radian) and\
                (math.fabs(dt1[format_av_sidx] - last_zone_first_point[format_av_sidx]) > self.izone_area_cond_s):
                  # start a new zone when
                  #   cond3: last is inv_line_overlap & distance exceeds certain range
                  print("[add] new zone as {} that contains inv_overlap zones".format(
                    (agent_idx, traj_idx, dt1[format_av_sidx])))
                  zone_from_points.append(ii)
                  last_zone_is_inv_zone = True

        agent_v0 = 0.0
        if full_iinfo_array.shape[0] > 0:
          agent_v0 = full_iinfo_array[0, format_agent_v0idx]

        # print("agent[{}] v0={}".format((agent_idx, traj_idx), agent_v0))
        # if (agent_idx == 370) or (agent_idx == 377):
        #   print("369 agent zone_from_points= ", (agent_idx, traj_idx), zone_from_points, agent_v0)
        
        # one trajectory of one agent contains several interaction zones
        ilen = full_iinfo_array.shape[0]
        num_1 = len(zone_from_points) - 1
        for zone_i, _ in enumerate(zone_from_points):
          # get interaction info: range_iinfo_array, for each interaction zone
          indexs = None
          if zone_i < num_1: # [zone_i, zone_i+1]
            indexs = list(
              range(zone_from_points[zone_i], zone_from_points[zone_i+1]))
          else:
            indexs = list(
              range(zone_from_points[zone_i], ilen))
          range_iinfo_array = full_iinfo_array[indexs, :]
          
          av_rear_occupied = ((agent_idx, traj_idx) in av_rear_already_occupied_dict.keys())
          av_front_occupied = ((agent_idx, traj_idx) in av_front_already_occupied_dict.keys())
          av_already_occupied = (av_rear_occupied or av_front_occupied)

          # # check special interaction relation [abandont]
          # # av's condition
          # zone_av_s_values = range_iinfo_array[:, format_av_sidx]
          # zone_agent_s_values = range_iinfo_array[:, format_agent_sidx]
          # if ((self.enable_judge_fpreempt) and\
          #     (av_already_occupied and (np.sum((zone_av_s_values < self.path_s_interval) * 1.0) > 1e-2)) == False) and\
          #     (zone_av_s_values.shape[0] >= 1):
          #   square_plan_v0 = (self.start_sva[1]**2)
          #   plan_max_dcc = self.search_acc_bounds[0]
          #   av_plan_s = zone_av_s_values[0]
          #   av_stop_s = zone_av_s_values[0] - self.path_s_interval
          #   agent_pred_s = zone_agent_s_values[0]
          #   av_stop_acc = -square_plan_v0 / (1e-3+2.0*av_stop_s)
          #   if av_stop_s <= 0.0:
          #     av_stop_acc = -1e+3
          #   max_dcc_square_vt = square_plan_v0 + 2.0*plan_max_dcc*av_plan_s
          #   max_dcc_plan_t = 1e+3
          #   if max_dcc_square_vt >= 0.0:
          #     max_dcc_plan_t = (math.sqrt(max_dcc_square_vt) - self.start_sva[1]) / plan_max_dcc
          #   ref_agent_cvm_t = agent_pred_s / max(agent_v0, self._cvm_protect_min_v)
          #   if (av_stop_acc < plan_max_dcc) and (max_dcc_plan_t < (ref_agent_cvm_t + self.yield_safe_time_gap)):
          #     # cond1: av can not stop before this interaciton zone
          #     # cond2: av can not slow down to wait agent pass first
          #     av_already_occupied = True
          #     print("DEBUG::SPECIAL ALREADY PREEMPT")

          # agent's condition
          agent_occupy_cond_dist = min(max(agent_v0 * 1.0, 0.1), self.path_s_interval)
          agent_already_occupied =\
            np.sum((range_iinfo_array[:, format_agent_sidx] < agent_occupy_cond_dist) * 1.0) > 1e-2
          
          # debug
          # if (agent_idx == 304625):
          #   print(">>> indexs", agent_idx, traj_idx)
          #   print("flags=", av_already_occupied, agent_already_occupied, (agent_idx, traj_idx) in av_already_occupied_dict.keys())
          #   # print(range_iinfo_array[:, format_av_sidx])
          #   print(range_iinfo_array[:, format_agent_sidx])

          # set relation
          set_relation = StiNode.relation_not_determined()

          if self.enable_judge_init_relation:
            conflicts = av_already_occupied and agent_already_occupied
            if conflicts:
              ColorPrinter.print('yellow', "Warning, conflict situation occurs, with {} and {}".format(
                (agent_idx, traj_idx), range_iinfo_array.shape))
              if av_rear_occupied:
                set_relation = StiNode.relation_influ()
                print(f"{agent_idx}/{traj_idx}/{zone_i} is with zone conflicts -> influ")
              elif av_front_occupied:
                set_relation = StiNode.relation_yield()
                print(f"{agent_idx}/{traj_idx}/{zone_i} is with zone conflicts -> react::yield")
              else:
                raise ValueError("Fatal value")
            elif av_already_occupied:
              if av_rear_occupied:
                set_relation = StiNode.relation_influ() # initially influ
                print(f"{agent_idx}/{traj_idx}/{zone_i} av rear -> influ")
                print("path av_s=", range_iinfo_array[:5, format_av_sidx])
              else: # av_front_occupied == true
                # set_relation = StiNode.relation_influ() # initially influ
                print(f"{agent_idx}/{traj_idx}/{zone_i} av front -> undo::ignored")
                # print("path av_s=", range_iinfo_array[:5, format_av_sidx])
            elif agent_already_occupied:
              set_relation = StiNode.relation_yield() # yield
              print(f"{agent_idx}/{traj_idx}/{zone_i} agent already occupied -> react::yield")
              # print("check", zone_i, num_1, ilen, zone_from_points)
              # print("path av_s=", range_iinfo_array[:5, format_av_sidx])

          # key [format]
          zone_key = (agent_idx, traj_idx, zone_i)
          if not zone_key in izone_key2points.keys():
            node_relation_index = int(StiNode.relation_index_bias() + relation_index)

            izone_key2points[zone_key] = []
            izone_key2node_relation_index[zone_key] = node_relation_index
            relation_index2izone_key[node_relation_index] = zone_key

            if math.fabs(set_relation - StiNode.relation_influ()) < 1e-3:
              init_influ_indexes.append(node_relation_index)
            if math.fabs(set_relation - StiNode.relation_yield()) < 1e-3:
              init_yield_indexes.append(node_relation_index)

            relation_index += 1

          for info in range_iinfo_array:
            izone_key2points[zone_key].append(int(len(izone_data_list)))

            izone_data_list.append(
              [zone_i, set_relation] + info.tolist()
            )

    # store to variables
    self.zone_seq_idx_idx :int = 0
    self.zone_relation_idx :int = 1
    self.zone_part_data_len :int = 2
    self.zone_full_data_len :int =\
      self.zone_part_data_len + InteractionFormat.iformat_len()

    izone_dt_array = np.array([]).reshape(0, self.zone_full_data_len)
    if len(izone_data_list) > 0:
      izone_dt_array = np.array(izone_data_list)

    # record interaction zone num and init stinode format
    self._izone_num :int = len(izone_key2points.keys())
    self.__format_node :StiNode= StiNode(izone_num=self._izone_num)

    self._node_key_s = self.__format_node.get_key_index('state_s')
    self._node_key_t = self.__format_node.get_key_index('state_t')
    self._node_key_v = self.__format_node.get_key_index('state_v')
    self._node_key_acc = self.__format_node.get_key_index('state_acc')
    self._node_key_pidx = self.__format_node.get_key_index('parent_index')
    self._node_key_sidx = self.__format_node.get_key_index('s_index')
    self._node_key_nidx = self.__format_node.get_key_index('node_index')
    self._node_key_flag = self.__format_node.get_key_index('leaf_flag')

    # record output values
    self.interaction_info = {
      # interaction zone data 
      'izone_data': izone_dt_array, # shape= zone_num x zone_info

      # dict of interaction zone data,
      #   map from zone_key=(agent_index, trajectory_index, zone_index),
      #   to relation index
      'izone_key2node_relation_index': izone_key2node_relation_index,
      'relation_index2izone_key': relation_index2izone_key,

      'init_influ_indexes': init_influ_indexes, # relation idnexes that initially already preempt
      "init_yield_indexes": init_yield_indexes, # relation idnexes that initially alraedy react
    }

  def _extract_range_iinfo(self, from_s: float, to_s: float) -> Dict:
    '''
    Extract range interaction informations
    :param from_s: s value from
    :param to_s: s value to
    :return: return {
      'has_interaction': bool, has interaction values or not
      'izone_data': np.ndarray, interaction zone array values, format follows izone_data in init_interaction_info()
    }
    '''
    izone_dt_array = self.interaction_info['izone_data'] # zone_num \times each_zone_info dimentions
    izone_key2node_relation_index = self.interaction_info['izone_key2node_relation_index']

    s_index :int= self.zone_part_data_len + InteractionFormat.iformat_index('av_s')
    agent_index :int= self.zone_part_data_len + InteractionFormat.iformat_index('agent_idx')
    traj_index :int= self.zone_part_data_len + InteractionFormat.iformat_index('agent_traj_idx')
    range_locs = np.logical_and(
      from_s <= izone_dt_array[:, s_index], 
      izone_dt_array[:, s_index] < (to_s + self.path_protect_s)) # <extended>

    # update variables
    range_izone_array = izone_dt_array[range_locs]
    has_interaction :bool= (range_izone_array.shape[0] > 0)

    # relation2ipoint_locs: record locations with same zone key
    #   one relation (intearction zone) may overlaps with multiple path points (named ipoint here).
    relation2ipoint_locs = {}
    relation_indexes = []
    for i, array in enumerate(range_izone_array):
      # range_izone_array: each interaction point > get the relation_index, and append it as list relation_indexes
      ai = int(array[agent_index])
      ti = int(array[traj_index])
      zi = int(array[self.zone_seq_idx_idx])
      zone_key = (ai, ti, zi)

      # node_relation_index: relation flag index in node representation
      node_relation_index = izone_key2node_relation_index[zone_key]
      relation_indexes.append(node_relation_index)

      if not node_relation_index in relation2ipoint_locs.keys():
        relation2ipoint_locs[node_relation_index] = []
      relation2ipoint_locs[node_relation_index].append(i)

    # protect s values in range (as they are <extended> at upper operation)
    area_protect_s_extended = range_izone_array[:, s_index] > to_s
    range_izone_array[area_protect_s_extended, s_index] = to_s

    # return values
    return {
      'has_interaction': has_interaction,
      'izone_data': range_izone_array,    # shape = (relation_len, state_num)
      'area_protect_s_extended': area_protect_s_extended,
      'relation_indexes': relation_indexes, # len = relation_len
      'relation2ipoint_locs': relation2ipoint_locs, # map relation_index > locations of ipoints
    }

  def __init_node_relations(self, parent_nodes: np.ndarray) -> None:
    '''
    Init parent_nodes' relations given range_iinfo_dict
    '''
    parent_nodes[:, self.interaction_info['init_influ_indexes']] = StiNode.relation_influ()
    parent_nodes[:, self.interaction_info['init_yield_indexes']] = StiNode.relation_yield()

  # def _extract_iinfo(self, traj_idxes: List) -> Dict:
  #   '''
  #   Extract interaction information along given trajectory
  #   :param traj_idxes: sequence of node index of the trajectory
  #   :return: return {
  #     'izone_data': np.ndarray, interaction zone array values, format follows izone_data in init_interaction_info()
  #     'plan_stva': np.ndarray, [s, t, v, a] values for the AV at interaction points (path)
  #   }
  #   '''
  #   izone_data = np.array([]).reshape(0, self.zone_full_data_len)
  #   stva_data = np.array([]).reshape(0, 4)

  #   format_av_sidx :int= InteractionFormat.iformat_index('av_s')
  #   s_index :int= self.zone_part_data_len + format_av_sidx
  #   node0 = StiNode(izone_num=self._izone_num)
  #   node1 = StiNode(izone_num=self._izone_num)
  #   for i, node_idx in enumerate(traj_idxes[:-1]):
  #     node0.update_values(self.node_records[node_idx])
  #     node1.update_values(self.node_records[traj_idxes[i+1]])

  #     get_dict = self._extract_range_iinfo(
  #       from_s=node0.get_state_value('state_s'), 
  #       child_node=node1.get_state_value('state_s'))

  #     if get_dict['has_interaction']:
  #       izone_data = np.concatenate(
  #         (izone_data, get_dict['izone_data']), axis=0)

  #       stva_data = np.concatenate((stva_data, 
  #         self._interpolate_node_given_s(
  #           node0, node1, inquiry_s=izone_data[:, s_index])
  #         ), axis=0)

  #   return {
  #     'izone_data': izone_data,
  #     'plan_stva': stva_data,
  #   }

  def __update_records(self, child_s_sample_index:int, child_nodes: np.ndarray, node_costs: np.ndarray) -> None:
    '''
    Update records of graph nodes and edges
    :param child_s_sample_index: s sample index of the child nodes
    :param child_nodes: child node array
    :param node_costs: cost for each child node
    '''
    assert not child_s_sample_index in self.layer_node_cost_record.keys(), "Unexpected error, pls check it"
    self.layer_node_cost_record[child_s_sample_index] = []

    for cnode, cost in zip(child_nodes, node_costs):
      pnode_idx = int(cnode[self._node_key_pidx])
      node_idx = int(cnode[self._node_key_nidx])
      # record layer nodes
      self.layer_node_cost_record[child_s_sample_index].append((node_idx, cost))

      # record child nodes
      self.node_records[node_idx] = cnode

      # record edges
      self.edge_pair_records[(pnode_idx, node_idx)] = cost

      # update goal record
      if (cnode[self._node_key_t] > self.goal_cond_tstamp) or\
         (cnode[self._node_key_v] <= self.stop_v_condition):
        key_tstamp = math.floor(
          (cnode[self._node_key_t] + self.goal_cond_reso * 0.5) / self.goal_cond_reso) * self.goal_cond_reso
        if not key_tstamp in self.goal_records:
          self.goal_records[key_tstamp] = []

        self.goal_records[key_tstamp].append([node_idx, cost])

    if child_nodes.shape[0] > 0:
      self.search_maximum_layer = max(self.search_maximum_layer, child_s_sample_index)
      self.search_max_speed = max(self.search_max_speed, np.max(child_nodes[:, self._node_key_v]))

  def _get_node_index(self, sidx: int, t_value: float, v_value: float, a_value: float) -> int:
    '''
    Return node index in s-v search space, where t, v values are checked inside the function.
    '''
    assert (0.0 <= t_value), "Error, input t value is out of bound" # (t_value <= self.search_horizon_T)
    assert (0.0 <= v_value) and (v_value < (self.max_v_limit + 1e-3)),\
      "Error, input v value is out of bound, with {}".format([v_value, self.max_v_limit])

    t_idx :int= math.floor((t_value + self.search_t_reso_2) / self.search_t_reso)
    v_idx :int= math.floor((v_value + self.search_ref_v_reso_2) / self.search_ref_v_reso)

    return (sidx * self.tv_space_grid_num +\
            t_idx * self.v_axis_grid_num + v_idx)

  def _get_node_indexes(self, s_values: np.ndarray, t_values: np.ndarray, v_values: np.ndarray) -> np.ndarray:
    '''
    Return node indexes (array like) in s-v search space, where only v is checked inside the function.
    '''
    check_v_values = np.sum(np.logical_or(v_values < 0.0, v_values > self.max_v_limit))
    assert check_v_values < 1e-6, "Error, has invalid v values, {}.".format(v_values)

    sidxs = np.round((s_values - self.s_sample_bias) / self.s_index_reso)
    tidxs = np.floor((t_values + self.search_t_reso_2) / self.search_t_reso)
    vidxs = np.floor((v_values + self.search_ref_v_reso_2) / self.search_ref_v_reso)

    return (sidxs * self.tv_space_grid_num +\
            tidxs * self.v_axis_grid_num + vidxs)

  def _interpolate_node_given_s(self, parent_node: StiNode, 
      child_node: StiNode, inquiry_s: np.ndarray) -> np.ndarray:
    '''
    Interpolate node value given s values
    :param parent_node: the node with child node to build up the graph edge in searching
    :param child_node: the node with parent node to build up the graph edge in searching
    :param inquiry_s: array like inquiry s values, which should be inside [parent_node.get_state_value('state_s'), child_node.get_state_value('state_s')]
    :return: s-t-v-a values given inquiry_s
    '''
    inquiry_move_s = inquiry_s - parent_node.get_state_value('state_s')
    stva = np.zeros((inquiry_s.shape[0], 4))

    # s = inquiry_s
    stva[:, 0] = inquiry_s
    # vt**2 = v0**2 + 2as
    square_vt = parent_node.get_state_value('state_v') **2 + 2.0*child_node.get_state_value('state_acc')*inquiry_move_s
    square_vt[square_vt < 0.0] = 0.0 # protect negative values
    stva[:, 2] = np.sqrt(square_vt)
    # t = t0 + s / (0.5*v0 + 0.5*vt))
    stva[:, 1] = parent_node.get_state_value('state_t') + \
      inquiry_move_s / (0.5*parent_node.get_state_value('state_v') + 0.5*stva[:, 2])
    # acc is constant
    stva[:, 3] = child_node.get_state_value('state_acc') # because using the constant acceleration model

    return stva

  def _interpolate_nodes_given_s(self, parent_node: np.ndarray, child_nodes: np.ndarray, inquiry_s: np.ndarray) -> np.ndarray:
    '''
    Interpolate node value given s values.
    :param parent_node: parent node of the graph edge in searching
    :param child_nodes: child nodes of the graph edge in searching
    :param inquiry_s: array like inquiry s values, values should inside [paren_node.s, np.max(child_nodes.s)]
    :return: [[s, t, v, a], ...] values given inquiry_s for [child0, child1, ...]
    '''
    ## The annotated operations are equal
    # inquiry_move_s = inquiry_s - parent_node[self._node_key_s]
    # inquiry_s_num :int= inquiry_s.shape[0]
    # seg_stvas = np.array([]).reshape((0, 4))
    # for cnode in child_nodes:
    #   stva = np.zeros((inquiry_s_num, 4))
    #   # s = inquiry_s
    #   stva[:, 0] = inquiry_s
    #   # vt**2 = v0**2 + 2as
    #   square_vt = parent_node[self._node_key_v] **2 + 2.0*cnode[self._node_key_acc]*inquiry_move_s
    #   square_vt[square_vt < 0.0] = 0.0 # protect negative values
    #   stva[:, 2] = np.sqrt(square_vt)
    #   # t = t0 + (move_s) / (0.5*v0 + 0.5*v1)
    #   stva[:, 1] = parent_node[self._node_key_t] + inquiry_move_s / (0.5*parent_node[self._node_key_v] + 0.5*stva[:, 2])
    #   # acc is constant
    #   stva[:, 3] = cnode[self._node_key_acc] # because using the constant acceleration model
    #
    #   seg_stvas = np.concatenate((seg_stvas, stva), axis=0)

    cnode_num :int= child_nodes.shape[0]
    inquiry_s_num :int= inquiry_s.shape[0]

    cache_move_s = np.repeat(np.array([inquiry_s]), cnode_num, axis=0) - parent_node[self._node_key_s]
    cache_cnodes_acc = np.repeat(
      np.array([child_nodes[:, self._node_key_acc]]), inquiry_s_num, axis=0).transpose()

    seg_stvas = np.zeros((cnode_num, inquiry_s_num, 4))
    # s = inquiry_s
    seg_stvas[:, :, 0] = parent_node[self._node_key_s] + cache_move_s
    # vt**2 = v0**2 + 2as
    cache_square_vt = parent_node[self._node_key_v]**2 + 2.0 * cache_cnodes_acc * cache_move_s
    cache_square_vt[cache_square_vt < 0.0] = 0.0 # protect negative values
    seg_stvas[:, :, 2] = np.sqrt(cache_square_vt)
    # t = t0 + s / (0.5*v0+0.5*v1)
    seg_stvas[:, :, 1] = parent_node[self._node_key_t] +\
      cache_move_s / (1e-6 + 0.5*parent_node[self._node_key_v] + 0.5*seg_stvas[:, :, 2])
    # acc is constant
    seg_stvas[:, :, 3] = cache_cnodes_acc

    return seg_stvas

  def _interpolate_node_given_t(self, parent_node: StiNode, 
      child_node: StiNode, inquiry_t: np.ndarray) -> np.ndarray:
    '''
    Interpolate node value given t values
    :param parent_node: the node with child node to build up the graph edge in searching
    :param child_node: the node with parent node to build up the graph edge in searching
    :param inquiry_t: array like inquiry s values, which should be inside [parent_node.state_t, child_node.state_t]
    :return: s-t-v-a values given inquiry_t
    '''
    inquiry_dts = inquiry_t - parent_node.get_state_value('state_t')
    stva = np.zeros((inquiry_dts.shape[0], 4))

    # t = inquiry_t
    stva[:, 1] = inquiry_t

    # acc is constant 
    stva[:, 3] = child_node.get_state_value('state_acc')

    # vt = v0 + a * t
    stva[:, 2] = parent_node.get_state_value('state_v') + child_node.get_state_value('state_acc') * inquiry_dts

    # s = s0 + v0 * t + 0.5 * a * t**2
    stva[:, 0] = parent_node.get_state_value('state_s') +\
                 parent_node.get_state_value('state_v') * inquiry_dts +\
                 0.5 * child_node.get_state_value('state_acc') * np.square(inquiry_dts)

    return stva

  def _interpolate_node_given_t_v2(self, parent_stva: np.ndarray, 
      child_stva: np.ndarray, inquiry_t: np.ndarray) -> np.ndarray:
    '''
    Interpolate node value given t values
    :param parent_stva: the node with child node to build up the graph edge in searching
    :param child_stva: the node with parent node to build up the graph edge in searching
    :param inquiry_t: array like inquiry s values, which should be inside [parent_stva.state_t, child_stva.state_t]
    :return: s-t-v-a values given inquiry_t
    '''
    inquiry_dts = inquiry_t - parent_stva[1]
    stva = np.zeros((inquiry_dts.shape[0], 4))

    # t = inquiry_t
    stva[:, 1] = inquiry_t

    # acc is constant 
    stva[:, 3] = child_stva[3]

    # vt = v0 + a * t
    stva[:, 2] = parent_stva[2] + child_stva[3] * inquiry_dts

    # s = s0 + v0 * t + 0.5 * a * t**2
    stva[:, 0] = parent_stva[0] +\
                 parent_stva[2] * inquiry_dts +\
                 0.5 * child_stva[3] * np.square(inquiry_dts)

    return stva

  def _get_leaf_node_locations(self, check_nodes: np.ndarray) -> np.ndarray:
    '''
    Return true when given node is a leaf node of the search tree
    :param check_nodes: nodes being checked
    '''
    # cond1: this s_layer is the last one
    # cond2: state t is out of search horion T
    # cond3: v is near 0

    leaf_node_locs = np.logical_or(
      check_nodes[:, self._node_key_flag] > 0.5, # leaf_node == 1.0
      check_nodes[:, self._node_key_sidx] >= self.s_maximum_layer_idx
    )
    leaf_node_locs = np.logical_or(
      leaf_node_locs,
      check_nodes[:, self._node_key_v] < self.stop_v_condition
    )

    return leaf_node_locs

  def _update_relations(self, 
      child_nodes: np.ndarray, range_iinfo_dict: Dict, new_relations: np.ndarray) -> None:
    # @note direct set the following equation fails to update values in child_nodes as they exists conflicts
    #   child_nodes[:, range_iinfo_dict['relation_indexes']] = new_relations

    # range_iinfo_dict['relation_indexes'], len   = edge_check_point_num, example = [14, 14, 14, 15, 15, 15, 16, 16, 16]
    # child_nodes:                          shape = (child_num, node_state_num)
    # new_relations                         shape = (child_num, edge_check_point_num)
    # valid_locs                            shape = (child_num, edge_check_point_num)
    map2relation_loc = range_iinfo_dict['relation_indexes']
    for child_i, relations in enumerate(new_relations):
      update_rela = {}
      for loc_i, rr in enumerate(relations):
        relation_loc = map2relation_loc[loc_i]
        if math.fabs(rr) > 0.3: # > 0.3 neglects undeter and ignored relations 
          if not relation_loc in update_rela:
            update_rela[relation_loc] = []
          update_rela[relation_loc].append(rr)

      for rela_loc_i, rela_list in update_rela.items():
        rela_list = np.unique(rela_list).tolist()
        assert len(rela_list) >= 1
        if len(rela_list) == 1:
          child_nodes[child_i, rela_loc_i] = rela_list[0] # update relation
        else:
          influ_in_list = (StiNode.relation_influ() in rela_list)
          preempt_in_list = (StiNode.relation_preempt() in rela_list)
          yield_in_list = (StiNode.relation_yield() in rela_list)
          
          conflict = preempt_in_list and yield_in_list
          if conflict:
            pass
          elif preempt_in_list and influ_in_list:
            # update relation
            child_nodes[child_i, rela_loc_i] = StiNode.relation_influ()
          elif yield_in_list and influ_in_list:
            # update relation
            child_nodes[child_i, rela_loc_i] = StiNode.relation_yield()

    # print("debug check updated relations")
    # print("relation_indexes=", range_iinfo_dict['relation_indexes'])
    # print("diff=", new_relations - child_nodes[:, range_iinfo_dict['relation_indexes']])

  @abc.abstractmethod
  def _edge_is_valid(self, range_iinfo_dict: Dict, 
      parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge parent node
    :param child_nodes: the edge child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0 # default all are valids (== 1.0), invalids == 0.0.

    return valid_costs

  # define function
  def __fill_record_tsv(self, record_index: int, p_stv0: Tuple, childs: np.ndarray) -> None:
    childs[:, self.__tsv_indexs[record_index, 0]] = p_stv0[1]
    childs[:, self.__tsv_indexs[record_index, 1]] = p_stv0[0]
    childs[:, self.__tsv_indexs[record_index, 2]] = p_stv0[2]

  def __interpolate_record_tsv(self, record_index: int, p_stv0: Tuple, check_t: float, childs: np.ndarray) -> None:
    update_locs = childs[:, self._node_key_t] >= check_t
    if np.sum(update_locs) > 0.0:
      # print(record_index, "debug update_locs", update_locs)
      # t = check_t; dt = t - t0;
      dts = check_t - p_stv0[1]
      square_dts = np.square(dts)
      childs[update_locs, self.__tsv_indexs[record_index, 0]] = check_t

      # s_t = s0 + v0*dt + 0.5*a*dt**2
      childs[update_locs, self.__tsv_indexs[record_index, 1]] = p_stv0[0] +\
        p_stv0[2]*dts + 0.5*childs[update_locs, self._node_key_acc]*square_dts

      # v_t = v0 + a*dt
      childs[update_locs, self.__tsv_indexs[record_index, 2]] =\
        p_stv0[2] + childs[update_locs, self._node_key_acc]*dts
      # print("tsvs", childs[update_locs, :][:, StiNode.tsv_record_indexs()])

  def _get_forward_simu_nodes(self, s_sample_index: int,
      parent_nodes: np.ndarray, parent_costs: np.ndarray, max_expand_num: int) -> Tuple:
    '''
    Get array of child nodes from parent nodes (all at one same s sample layer).
    :param s_sample_index: s sample index of the parent nodes
    :param parent_nodes: the parent nodes
    :param parent_costs: accumulated costs to the parent nodes
    :param max_expand_num: maximum forward expansion number for each parent node 
    :return: child_nodes, edge_costs
    '''
    empty_child_nodes = np.array([]).reshape(0, self.__format_node.len_list_values())
    empty_costs = np.array([])

    ## return when out of search range
    nxt_s_sample_index = s_sample_index + 1
    if (nxt_s_sample_index > self.s_maximum_layer_idx):
      # return 0 child nodes when out of search layer horizon
      return empty_child_nodes, empty_costs

    s0 = self.s_samples[s_sample_index]
    s1 = self.s_samples[nxt_s_sample_index]
    range_iinfo_dict = self._extract_range_iinfo(from_s=s0, to_s=s1)

    ## return when parent nodes size == 0
    if parent_nodes.shape[0] == 0:
      # return 0 child nodes when parent nodes num == 0
      return empty_child_nodes, empty_costs

    ## return when parent nodes all are leaf nodes
    cache_parent_nodes = parent_nodes
    if not (s_sample_index == 0):
      # disable to expand leaf nodes for all not start nodes
      leaf_locs = self._get_leaf_node_locations(parent_nodes)
      cache_parent_nodes = parent_nodes[np.logical_not(leaf_locs)]

    if cache_parent_nodes.shape[0] == 0:
      # return 0 child nodes when non-leaf parent nodes num == 0
      return empty_child_nodes, empty_costs

    ## forward simulation
    # preparation
    seg_s = s1 - s0
    v1_limit = self.v_limits[nxt_s_sample_index]
    v0s = cache_parent_nodes[:, self._node_key_v]
    v0s_square = np.square(v0s)

    # get v1 lower bound and upper bound for each parent node
    cache = v0s_square + 2.0 * self.search_acc_bounds[0] * seg_s
    cache[cache <= 0.0] = 0.0
    v1_lbounds = np.sqrt(cache)
    cache = v0s_square + 2.0 * self.search_acc_bounds[1] * seg_s
    v1_ubounds = np.sqrt(cache)
    v1_ubounds[v1_ubounds > v1_limit] = v1_limit

    v1_bounds = np.vstack((v1_lbounds, v1_ubounds)).transpose()
    # print("layer::forward_simu_debug")
    # print("layer_from_to=", s_sample_index, nxt_s_sample_index)

    def gen_child_nodes(pnode: np.ndarray, pnode_cost: float, cnode_vl: float, cnode_vu: float) -> Tuple:
      '''
      Generate child nodes
      :return tuple of child_nodes, action_costs
      '''
      v_sample_range = cnode_vu - cnode_vl
      if v_sample_range < 0.0:
        return empty_child_nodes, empty_costs

      ## generate child_samples
      t0 = pnode[self._node_key_t]
      v0 = pnode[self._node_key_v]
      a0 = pnode[self._node_key_acc]

      v_sample_num = min(
        round(v_sample_range / self.search_ref_v_reso), max_expand_num)
      v_sample_num = max(v_sample_num, 2)
      v1_samples = np.linspace(cnode_vl, cnode_vu, num=v_sample_num)

      # try add one v1 sample with same value with parent node (making the curve smooth)
      square_dist2v0 = np.square(v1_samples - v0)
      min_loc = np.argmin(square_dist2v0)
      min_dist2v0 = math.sqrt(np.min(square_dist2v0))
      if v0 > 1e-1: # only when v0 is not 0, try add v0
        if min_dist2v0 <= self.search_ref_v_reso:
          v1_samples[min_loc] = v0 # set value as v0
        else:
          v1_samples = np.array(v1_samples.tolist() + [v0]) # add v0 sample
      else:
        v1_samples = v1_samples[v1_samples > 1e-2] # remove v=0 child node
      v_sample_num = v1_samples.shape[0] # update v sample num

      child_samples = np.repeat(np.array([pnode.tolist()]), v_sample_num, axis=0) # copy parent node data
      child_samples[:, self._node_key_pidx] = pnode[self._node_key_nidx] # parent_idx
      child_samples[:, self._node_key_sidx] = nxt_s_sample_index # child_s_idx
      child_samples[:, self._node_key_s] = s1 # child_s
      child_samples[:, self._node_key_v] = v1_samples # child_v
      child_samples[:, self._node_key_acc] = (np.square(v1_samples) - v0**2) / (2.0 * seg_s) # child_acc

      # t1= t0 + seg_s / (0.5*v0 + 0.5*v1)
      child_samples[:, self._node_key_t] = t0 + seg_s / (0.5*v0 + 0.5*v1_samples)
      # print("gen::debug0", child_samples.shape)

      ## remove & truncate child_samples
      # remove jerk invalid ones
      get_abs_jerks = np.abs(
        (child_samples[:, self._node_key_acc] - pnode[self._node_key_acc]) \
          / (1e-4 + child_samples[:, self._node_key_t] - t0)
        )
      child_samples = child_samples[get_abs_jerks <= self.search_jerk_limit]
      # print("gen::debug1", child_samples.shape)

      # truncate t_horizon if out of search bound
      out_bound_locs = child_samples[:, self._node_key_t] > self.search_horizon_T
      out_bound_samples = child_samples[out_bound_locs]
      if out_bound_samples.shape[0] > 0:
        child_samples[out_bound_locs, self._node_key_t] = self.search_horizon_T
        delta_ts = self.search_horizon_T - t0

        # update v1 values [5]: vt = v0 + a*t
        child_samples[out_bound_locs, self._node_key_v] =\
          v0 + out_bound_samples[:, self._node_key_acc] * delta_ts
        # update s1 values [3]: s = s0 + v0*t + 0.5*a*t**2
        child_samples[out_bound_locs, self._node_key_s] = s0 +\
          v0 * delta_ts + 0.5 * out_bound_samples[:, self._node_key_acc] * np.square(delta_ts)
        # update leaf_node_flag [7]
        child_samples[out_bound_locs, self._node_key_flag] = 1.0

      # print("gen::debug2", child_samples.shape)
      # update child tsv values: during planning (assume 2.0s should show intention)
      stv0 = (s0, t0, v0)
      # t0 = 0.0 is recorded at index = 0, 
      if (t0 < 0.5):
        self.__interpolate_record_tsv(1, stv0, check_t=0.5, childs=child_samples)
      # if (t0 < 1.0):
      #   self.__interpolate_record_tsv(2, stv0, check_t=1.0, childs=child_samples)
      # if (t0 < 2.0):
      #   self.__interpolate_record_tsv(3, stv0, check_t=2.0, childs=child_samples)
      # if (t0 < 3.0):
      #   self.__interpolate_record_tsv(4, stv0, check_t=3.0, childs=child_samples)

      # remove invalid childs and get interaction costs
      valid_and_costs = self._edge_is_valid(
        range_iinfo_dict=range_iinfo_dict, 
        parent_node=pnode, child_nodes=child_samples)
      valid_locs = valid_and_costs[:, 0] > 0.5 # valid == 1.0
      # print("gen::debug3", child_samples.shape)
      # print(child_samples[:, self._node_key_v])
      # print(valid_locs)

      ## final update indexes and costs
      child_samples[:, self._node_key_nidx] =\
        self._get_node_indexes(s_values=child_samples[:, self._node_key_s], 
          t_values=child_samples[:, self._node_key_t], v_values=child_samples[:, self._node_key_v]) # child_idx

      # remove whose parent node index == self: generally happens among truncated nodes
      valid_locs = np.logical_and(
        valid_locs,
        np.abs(child_samples[:, self._node_key_nidx] - pnode[self._node_key_nidx]) > 0.5
      )

      ## calculate costs: parent_cost + action_cost + interaction_cost (edge_cost)
      child_samples = child_samples[valid_locs]
      valid_and_costs = valid_and_costs[valid_locs]
      # print("gen::debug4", child_samples.shape)
      # print(" ")

      costs = pnode_cost + \
        self.__cost_metric.get_action_costs(
          parent_node=pnode, child_nodes=child_samples, child_v_limit=cnode_vu) +\
        valid_and_costs[:, 1]

      return child_samples, costs

    # for each parent node, get child nodes and concatenate them
    cache_valid_nodes :np.ndarray= np.array([]).reshape(0, self.__format_node.len_list_values())
    get_costs :np.ndarray= np.array([])

    get_list = [gen_child_nodes(pnode, pnode_cost, vbound[0], vbound[1]) \
      for pnode, pnode_cost, vbound in zip(cache_parent_nodes, parent_costs, v1_bounds)]
    for  valid_child_array, costs in get_list:
      cache_valid_nodes = np.concatenate((cache_valid_nodes, valid_child_array), axis=0)
      get_costs = np.concatenate((get_costs, costs))
    # print("gen::debug f=", cache_valid_nodes.shape)

    # for child nodes with same node index, remain the one with lowest cost
    cache_dict = {}
    for i, node, cost in zip(range(0, cache_valid_nodes.shape[0]), cache_valid_nodes, get_costs):
      nid = int(node[self._node_key_nidx])
      if not nid in cache_dict:
        cache_dict[nid] = (i, cost)
      elif cost < cache_dict[nid][1]:
        cache_dict[nid] = (i, cost)

    child_nodes = np.array([cache_valid_nodes[dt[0]].tolist() for _, dt in cache_dict.items()])
    child_costs = np.array([get_costs[dt[0]] for _, dt in cache_dict.items()])

    ## records
    self.__update_records(child_s_sample_index=nxt_s_sample_index, 
      child_nodes=child_nodes, node_costs=child_costs)

    return child_nodes, child_costs

  def get_path2node(self, goal_node_index: int, max_iterations: int=50) -> Tuple:
    '''
    Calculate list of nodes, the path to goal node index
    :param goal_node_index, the node index of the goal
    :param maximum iterations to extract the path
    :return: extracted_success, list of nodes
    '''
    if not goal_node_index in self.node_records:
      ColorPrinter.print('yellow', "get_path2node::warning, node={} is not recorded".format(goal_node_index))
      return False, []

    success = False
    get_node_sequence = []

    current_node :int= goal_node_index
    for i in range(max_iterations):
      get_node_sequence.append(copy.copy(current_node))

      parent_node = int(self.node_records[current_node][self._node_key_pidx])
      if parent_node == -1:
        success = True
        break
      current_node = parent_node

    if success == False:
      get_node_sequence = []
    return success, get_node_sequence

  @abc.abstractmethod
  def start_searching(self) -> None:
    '''
    Searching for valid speed profiles
    :param start_sva: initial s,v,a values of AV: corresponding to self.s_samples[0]
    '''
    start_sva = self.start_sva
    assert math.fabs(start_sva[0] - self.s_samples[0]) < 1e-3, "Error, s values unmatched {}.".format(
      [start_sva[0], self.s_samples[0]])

    self.reinit()

    # forcibly correct input acc (initial state may out of search bounds)
    start_sva[2] = max(start_sva[2], self.search_acc_bounds[0])
    start_sva[2] = min(start_sva[2], self.search_acc_bounds[1])

    max_expand_num = self.search_max_v_sample_num

    # init with start node
    start_node = StiNode(izone_num=self._izone_num)
    start_node.set_state_value('parent_index', -1)
    start_node.set_state_value('s_index', 0)
    start_node.set_state_value('node_index', self._get_node_index(
      sidx=0, t_value=0., v_value=start_sva[1], a_value=start_sva[2]))
    start_node.set_state_value('state_t', 0.0)
    start_node.set_state_value('state_s', start_sva[0])
    start_node.set_state_value('state_v', start_sva[1])
    start_node.set_state_value('state_acc', start_sva[2])

    self.layer_node_cost_record[start_node.s_index()] = [(start_node.node_index(), 0.0)]
    self.node_records[start_node.node_index()] = start_node.get_array_of_values() # update start node record

    from_s_sample_index :int= start_node.s_index()
    from_layer_nodes = np.array([start_node.get_list_of_values()])
    from_layer_node_costs = np.array([0.0])

    self.__fill_record_tsv(
      record_index=0, p_stv0=(start_sva[0], 0.0, start_sva[1]), childs=from_layer_nodes)
    self.__init_node_relations(from_layer_nodes)

    # forward expansion
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("start_searching::debug()")
    for s_sample_i in range(1, self.s_maximum_layer_idx+1):
      nxt_layer_nodes, nodes_costs =\
        self._get_forward_simu_nodes(
          s_sample_index=from_s_sample_index, 
          parent_nodes=from_layer_nodes, parent_costs=from_layer_node_costs, 
          max_expand_num=max_expand_num)

      print(from_s_sample_index, "to", s_sample_i, 
            "::layer=", from_layer_nodes.shape, nxt_layer_nodes.shape)
      if nxt_layer_nodes.shape[0] == 0:
        break # no more valid nodes

      from_s_sample_index = s_sample_i
      from_layer_nodes = nxt_layer_nodes.copy()
      from_layer_node_costs = nodes_costs.copy()

      # remove leaf nodes & speed==0.0 nodes for next round expansion
      speed_0_locs = from_layer_nodes[:, self._node_key_v] < 1e-3
      leaf_node_locs = from_layer_nodes[:, self._node_key_flag] > 0.5 # leaf_node == 1.0

      invalid_locs = np.logical_or(speed_0_locs, leaf_node_locs)
      valid_locs = np.logical_not(invalid_locs)

      from_layer_nodes = from_layer_nodes[valid_locs]
      from_layer_node_costs = from_layer_node_costs[valid_locs]

  def _extract_goal_node_candidates(self) -> Tuple[bool, List]:
    '''
    Return goal layer index
    :return: has_valid_goal_layer, key_tstamp
    '''
    # collect layer key and data amount
    key_ts = [float(key_value) for key_value in sorted(self.goal_records.keys(), key=lambda p : float(p), reverse=True)]
    key_nums = [len(self.goal_records[kk]) for kk in key_ts]
    len_key_ts = len(key_ts)
    if len_key_ts == 0:
      # no solution, return false
      print("_extract_goal_node_candidates::warning(), key tstamp num == 0")
      return False, []

    # chose a layer key according to certain human design rule
    chose_key_t = key_ts[0] # default chose first key
    for i, _ in enumerate(key_ts[:-1]):
      num_key = float(len(self.goal_records[key_ts[i]]))
      num_nxt_key = float(len(self.goal_records[key_ts[i+1]]))
      if (num_key > (0.5 * num_nxt_key)) and (num_key > 10):
        chose_key_t = key_ts[i]
        break

    # extract certain number of node candidates
    # TODO(abing): enable goal candidates after chose_key_t
    node_candidates = self.goal_records[chose_key_t]
    node_candidates = sorted(node_candidates, key=lambda n: n[1]) # cost from low to high

    return True, node_candidates

  def get_braking_stop_along_the_path(self, interval_t: float) -> np.ndarray:
    '''
    Return stva array of the trajectory when av braking to stop along the given path
    '''
    ref_dcc = self.search_acc_bounds[0]
    s0 = self.start_sva[0]
    v0 = self.start_sva[1]
    v0_square = v0**2

    pnode_stva = [s0, 0.0, v0, self.start_sva[2]]
    stva_array = [pnode_stva]

    for s in self.ori_s_samples[1:]:
      ds = s - s0
      # vt = sqrt(v0**2 + 2.0*a*s)

      dd = v0_square + 2.0*ref_dcc*ds
      if dd >= 0.0:
        vt = math.sqrt(dd)
        # vt = v0 + a*t
        t = (vt - v0) / ref_dcc
        stva_array.append([s, t, vt, ref_dcc])
      else:
        vt = 0.0
        t = (vt - v0) / ref_dcc
        s = (-v0**2) / (2.0 * ref_dcc)
        stva_array.append([s, t, vt, ref_dcc])
        break
    
    traj_stva_array = []
    stva_array = np.array(stva_array)
    len_2 = stva_array.shape[0] - 2
    for i in range(1, stva_array.shape[0]):
      cnode_stva = stva_array[i, :]

      from_t = math.ceil(pnode_stva[1] / interval_t) * interval_t
      to_t = math.ceil(cnode_stva[1] / interval_t) * interval_t - 1e-3
      if i == len_2:
        to_t += 2e-3 # sample last t value
      inquiry_t = np.arange(from_t, to_t, interval_t)
      seg_stva = self._interpolate_node_given_t_v2(pnode_stva, cnode_stva, inquiry_t)

      traj_stva_array = traj_stva_array + seg_stva.tolist()

      pnode_stva = cnode_stva.copy()

    traj_stva_array = np.array(traj_stva_array)

    return traj_stva_array

  @abc.abstractmethod
  def get_planning_results(self, interval_t: float, set_result_max_num: int=None) -> Dict:
    '''
    Select valid trajectory after generating_trajectories()
    :param interval_t: output trajectory's interval time
    :param set_result_max_num: maximum planning result amount
    :return: return dict of trajectory, following formats as: {
      'has_result': bool,
      'stva_array': np.ndarray,
      'sum_action_cost': float,
      'sum_react_cost': float,
    }
    '''
    get_result = {
      'has_result': False,
      'stva_array': None,
      'sum_action_cost': 0.,
      'sum_react_cost': 0.,
    }
    self.__rviz_plan_results = []

    if set_result_max_num is None:
      set_result_max_num = self.result_traj_max_num
    
    # extract certain number of goal node candidates
    is_valid, node_candidates = self._extract_goal_node_candidates()
    if is_valid == False:
      return get_result

    cache_plan_results :List= []
    for node_index, node_cost in node_candidates:
      flag, node_seq = self.get_path2node(goal_node_index=int(node_index))
      if flag == False:
        continue
      node_seq.reverse()

      traj_stva_array = []
      speed_limit_array = []
      sum_action_cost = 0.
      sum_react_cost = 0.

      len_1 = len(node_seq) - 1
      len_2 = len(node_seq) - 2
      pnode = StiNode(izone_num=self._izone_num)
      cnode = StiNode(izone_num=self._izone_num)
      for i in range(0, len_1):
        pnode.update_values(self.node_records[node_seq[i]])
        cnode.update_values(self.node_records[node_seq[i+1]])

        # stva array: interpolate according to t axis
        from_t = math.ceil(pnode.get_state_value('state_t') / interval_t) * interval_t
        to_t = math.ceil(cnode.get_state_value('state_t') / interval_t) * interval_t - 1e-3
        if i == len_2:
          to_t += 2e-3 # sample last t value
        inquiry_t = np.arange(from_t, to_t, interval_t)
        seg_stva = self._interpolate_node_given_t(pnode, cnode, inquiry_t)
        seg_stva[seg_stva[:, 2] < 0.0, 2] = 0.0 # correct negative speeds

        traj_stva_array = traj_stva_array + seg_stva.tolist()

        # v_limit array
        pvlimit = self.v_limits[pnode.s_index()]
        cvlimit = self.v_limits[cnode.s_index()]
        seg_ds = cnode.get_state_value('state_s') - pnode.get_state_value('state_s')

        cache = (seg_stva[:, 0] - pnode.get_state_value('state_s')) / max(1e-2, seg_ds)
        seg_vlimits = (1.0 - cache) * pvlimit + cache * cvlimit
        speed_limit_array = speed_limit_array + seg_vlimits.tolist()

        # sum_action_cost calculation
        sum_action_cost += self.__cost_metric.get_action_cost(pnode, cnode)

      speed_limit_array = np.array(speed_limit_array)
      if len(traj_stva_array) > 0:
        cache_plan_results.append({
          'traj_idxes': node_seq,
          'traj_stva': np.array(traj_stva_array),
          'speed_limits': speed_limit_array,
          'sum_action_cost': sum_action_cost, 
          'sum_react_cost': sum_react_cost,
        })

      if len(cache_plan_results) >= set_result_max_num:
        break

    self.__rviz_plan_results = cache_plan_results
    # pick out the cost-min one as output trajectory
    if len(cache_plan_results) > 0:
      cache = cache_plan_results[0] # return the first / lowest cost one

      get_result = {
        'has_result': True,
        'stva_array': cache['traj_stva'],
        'speed_limits': speed_limit_array,
        'sum_action_cost': cache['sum_action_cost'],
        'sum_react_cost': cache['sum_react_cost'],
        'edge_counts': len(self.edge_pair_records.keys()),
      }

    return get_result

  def visualize_searching(self) -> None:
    '''
    Visualize the search process
    '''
    if self.enable_debug_rviz:
      import matplotlib
      import matplotlib.pyplot as plt
      import matplotlib.colors as mcolors
      import paper_plot.utils as plot_utils
      import paper_plot.functions as plot_funcs
      s_t_label_loc = 'lower right'
      s_v_label_loc = 'lower right'

      # enables
      replace_labels = False

      # preparations
      format_aidx :int= InteractionFormat.iformat_index('agent_idx')
      format_sidx :int= InteractionFormat.iformat_index('av_s')
      format_tidx :int= InteractionFormat.iformat_index('agent_t')
      format_vidx :int= InteractionFormat.iformat_index('agent_v')

      pred_data = self.interaction_info['izone_data'][:, self.zone_part_data_len:]
      color_dict = mcolors.XKCD_COLORS 
      color_keys = list(color_dict.keys())
      color_num: int = len(color_dict)
      icolor = [color_dict[color_keys[int(dta[format_aidx]) % color_num]] for dta in pred_data]

      icollisions_zorder = 8.0
      imarkertype = 's'
      edge_info_dicts = [
        [self.edge_pair_records, '-', 'edges', 'grey']
      ]

      trajs_cmap=plt.cm.get_cmap('winter')
      color_bar_pad = 0.1
      cmp_norm = matplotlib.colors.BoundaryNorm(
        np.arange(0.0, 1.01, 0.05), trajs_cmap.N)

      param_edge_width = 0.25

      costs_list = np.array([(dt['sum_action_cost'] + dt['sum_react_cost']) for dt in self.__rviz_plan_results])
      traj_cost_bias = 0.0
      traj_cost_max = 1.0
      if costs_list.shape[0] > 0:
        traj_cost_bias = np.min(costs_list)
        traj_cost_max = np.max(costs_list) - traj_cost_bias

      # TODO rviz cmd traj
      # cmd_plan_idxes = self.__rviz_plan_results[-1]['traj_idxes']
      # cmd_plan_stva = self.plan_final_result['stva_array']
      # dict_cache = self.__extract_trajectory_ipoints_values(cmd_plan_idxes)
      # demo_plan_izones = dict_cache['izones']
      # demo_plan_iplan_stva = dict_cache['plan_stva']

      ##############################################################
      plot_fig = plt.figure()
      plot_utils.fig_reset()

      def get_s_label(i) -> str:
        if i <= 1:
          return "$s_{}$".format(i)
        elif i == 2:
          return "..."
        elif i == 4:
          return "$s_k$"
        return ""
      def get_v_label(v) -> str:
        if math.fabs(v) < 1e-3:
          return "0"
        return ""     

      ##############################################################
      fig1_ax = plot_fig.add_subplot(131)
      plot_utils.subfig_reset()
      plot_utils.axis_set_title(fig1_ax, 'time stamp, 0-max: blue-red')
      self.ispace.plot_av_path_and_agent_predicitons(plot_axis=fig1_ax)

      ##############################################################
      # fig4_ax = plot_fig.add_subplot(132)
      # plot_utils.subfig_reset()
      # plot_utils.axis_set_title(fig1_ax, 'polygons')
      # self.ispace.plot_av_and_agent_polygons(plot_axis=fig4_ax)

      ##############################################################
      # figure1: s-t graph
      fig3_ax = plot_fig.add_subplot(132)
      plot_utils.subfig_reset()
      plot_utils.fig_reset()

      ## plot start node/ edges
      first_initial = True
      initial_node = None
      s_t_max_s_value = 0.0
      parent_node = StiNode(izone_num=self._izone_num)
      child_node = StiNode(izone_num=self._izone_num)
      for edge_dict, edge_marker, edge_label, edge_color in edge_info_dicts:
        first_edge = True
        for idx_pair, _ in edge_dict.items():
          # print("rviz", parent_node_idx, child_node_idx)
          parent_node_idx = idx_pair[0]
          child_node_idx = idx_pair[1]
          
          parent_node.update_values(self.node_records[parent_node_idx])
          child_node.update_values(self.node_records[child_node_idx])

          if first_initial and (parent_node.get_state_value('state_t') < 1e-6):
            initial_node = parent_node
            plt.plot(initial_node.get_state_value('state_s'), initial_node.get_state_value('state_t'), 'or', label='initial state')
            plt.legend(loc=s_t_label_loc)

            first_initial = False

          s_t_max_s_value = max(s_t_max_s_value, child_node.get_state_value('state_s'))
          xys = np.array([[parent_node.get_state_value('state_s'), parent_node.get_state_value('state_t')], 
            [child_node.get_state_value('state_s'), child_node.get_state_value('state_t')]])
          if first_edge:
            plt.plot(xys[:, 0], xys[:, 1], edge_marker, color=edge_color, 
                     lw=param_edge_width, label=edge_label)
            plt.legend(loc=s_t_label_loc)
            first_edge = False
          else:
            plt.plot(xys[:, 0], xys[:, 1], edge_marker, color=edge_color, lw=param_edge_width)

      ## plot s-t values in their predictions
      plt.scatter(pred_data[:, format_sidx], pred_data[:, format_tidx], color=icolor, 
        marker=imarkertype, label="predictions", zorder=icollisions_zorder)
      plt.legend(loc=s_t_label_loc)

      ## plot result trajs
      if self.__enable_plot_plan_results:
        for tid, traj_result in enumerate(self.__rviz_plan_results):
          traj_stva = traj_result['traj_stva']
          plt.plot(traj_stva[:, 0], traj_stva[:, 1], 
            color=trajs_cmap((costs_list[tid] - traj_cost_bias) / (1e-3+traj_cost_max))
            )

      ## plot axis labels
      plot_utils.axis_set_xlabel(fig3_ax, "s (m)")
      plot_utils.axis_set_ylabel(fig3_ax, "t (s)")

      x_ticks = self.s_samples
      xlim_range = (-2.0, s_t_max_s_value + 5.0)
      ylim_range = (-1.0, self.search_horizon_T + 1.0)
      if replace_labels:
        x_labels = [get_s_label(i) for i, _ in enumerate(x_ticks)]
        plt.xticks(x_ticks, x_labels) 
      plt.xlim(xlim_range)

      if replace_labels:
        y_ticks = list(range(0, 8, 2))
        plt.yticks(y_ticks, [get_v_label(t) for t in y_ticks])
      plt.ylim(ylim_range)

      plt.grid(color='k', linestyle='-.')
      # plt.colorbar(
      #   matplotlib.cm.ScalarMappable(norm=cmp_norm, cmap=trajs_cmap),
      #   shrink=1.0, pad=color_bar_pad, orientation='vertical')
      
      ##############################################################
      # figure2: s-v graph
      fig2_ax = plot_fig.add_subplot(133)
      plot_utils.subfig_reset()

      ## plot speed limits
      plt.plot(self.s_samples, self.v_limits, '^g', label='speed upper bounds')
      plt.legend(loc=s_v_label_loc)
      
      ## plot start node/ edges
      first_initial = True
      initial_node = None

      parent_node = StiNode(izone_num=self._izone_num)
      child_node = StiNode(izone_num=self._izone_num)
      for edge_dict, edge_marker, edge_label, edge_color in edge_info_dicts:
        first_edge = True
        for idx_pair, _ in edge_dict.items():
          # print("rviz", parent_node_idx, child_node_idx)
          parent_node_idx = idx_pair[0]
          child_node_idx = idx_pair[1]
          parent_node.update_values(self.node_records[parent_node_idx])
          child_node.update_values(self.node_records[child_node_idx])
          
          if first_initial and (parent_node.get_state_value('state_t') < 1e-6):
            initial_node = parent_node
            plt.plot(initial_node.get_state_value('state_s'), initial_node.get_state_value('state_v'), 'or', label='initial state')
            plt.legend(loc=s_v_label_loc)

            first_initial = False

          xys = np.array([[parent_node.get_state_value('state_s'), parent_node.get_state_value('state_v')], 
            [child_node.get_state_value('state_s'), child_node.get_state_value('state_v')]])
          if first_edge:
            plt.plot(xys[:, 0], xys[:, 1], edge_marker, color=edge_color, 
                     lw=param_edge_width, label=edge_label)
            plt.legend(loc=s_v_label_loc)
            first_edge = False
          else:
            plt.plot(xys[:, 0], xys[:, 1], edge_marker, color=edge_color, 
                     lw=param_edge_width)

      ## plot agents s-v values in their predictions
      plt.scatter(pred_data[:, format_sidx], pred_data[:, format_vidx], color=icolor, 
        marker=imarkertype, label="predictions", zorder=icollisions_zorder)
      plt.legend(loc=s_v_label_loc)

      ## plot result trajs
      if self.__enable_plot_plan_results:
        for tid, traj_result in enumerate(self.__rviz_plan_results):
          traj_stva = traj_result['traj_stva']
          plt.plot(traj_stva[:, 0], traj_stva[:, 2], 
            color=trajs_cmap((costs_list[tid] - traj_cost_bias) / (1e-3+traj_cost_max))
            )

      ## plot axis labels
      plot_utils.axis_set_xlabel(fig2_ax, "s (m)")
      plot_utils.axis_set_ylabel(fig2_ax, "v (m/s)")

      ylim_range = (-1.0, self.search_max_speed + 5.0)
      if replace_labels:
        x_ticks = self.s_samples
        x_labels = [get_s_label(i) for i, _ in enumerate(x_ticks)]
        plt.xticks(x_ticks, x_labels) 
      plt.xlim(xlim_range)

      if replace_labels:
        y_ticks = list(range(0, 32, 2))
        plt.yticks(y_ticks, [get_v_label(v) for v in y_ticks])
      plt.ylim(ylim_range)

      plt.grid(color='k', linestyle='-.')
      plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=cmp_norm, cmap=trajs_cmap),
        shrink=1.0, pad=color_bar_pad, orientation='vertical')
      
      plt.show()
