import abc
from typing import Dict, List, Tuple
import copy
import numpy as np
import time
import math

from envs.navigation_map import NavigationMap
from envs.commonroad.simulation.utility import limit_agent_length_width
from utils.transform import XYYawTransform
from type_utils.agent import EgoAgent
from type_utils.trajectory_point import TrajectoryPoint
from planners.interaction_space import InteractionSpace
from planners.lattice_planner.lattice_planner import LatticePlanner
from pymath.curves.bspline import SplineCurve
from shapely.geometry import Polygon

from planners.st_search.zone_stv_search import ZoneStvGraphSearch
from planners.st_search.zone_stv_search_ca import ZoneStvGraphSearchCollisionAvoidance
from planners.st_search.zone_stv_search_izone_ca import ZoneStvGraphSearchIZoneCollisionAvoidance

from planners.st_search.zone_stv_search_izone_relations import ZoneStvGraphSearchIZoneRelations

from utils.colored_print import ColorPrinter

TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE

class BasicPlanner:
  '''
  Motion planning algorithm basic model.
  '''
  def __init__(self, planner_type: str,
                     plan_horizon_T: float, 
                     predict_traj_mode_num: int,
                     planning_dt: float=0.1,
                     avoid_d_range: List= [-2.0, 2.0],
                     space_reso: float=0.2,
                     prediction_dt: float=0.5,
                     cost_cofficients: Tuple[float, float, float, float]= (0.1, 0.5, 0.2, 0.2),
                     change_hebavior_init_cost: float= 1.0,
                     change_behavior_cost_decay: float= 0.9,
                     reaction_config: Dict= None):
    '''
    :param planner_type: planner type, ['izone_rela']
    :param plan_horizon_T: plan horizon (seconds)
    :param avoid_d_range: lateral avoidance range for behavior planning (meters)
    :param space_reso: resolution of x y space for discretizing
    :param planning_dt: plan horizon delta t
    :param prediction_dt: prediction trajectory interval time stamp
    :param cost_cofficients: cost coefficient for trajectory selection, including
                             [path_cost, horizon_dist, action_cost, reaction_cost]
    :param change_hebavior_init_cost: extra cost if the behavior need to be changed
    :param change_behavior_cost_decay: decay of the extra change_hebavior_init_cost
    '''
    self.planner_type = planner_type
    self.plan_horizon_T = plan_horizon_T
    self.predict_traj_mode_num = predict_traj_mode_num
    self.space_reso = space_reso
    self.planning_dt = planning_dt
    self.prediction_dt = prediction_dt
    self.cost_cofficients = cost_cofficients

    self.last_behavior_type = None
    self.change_hebavior_init_cost = change_hebavior_init_cost
    self.change_behavior_cost_decay = change_behavior_cost_decay
    self.change_behavior_cost = change_hebavior_init_cost

    self.reaction_config = reaction_config

    self.__planners_ignoring_agent_rears = [
      'ca+',
      'izone_ca+',
      'izone_rela+',
    ]

    self.constraints = {
      'a_min': -6,  # Minimum feasible acceleration of vehicle
      'a_max': 6,   # Maximum feasible acceleration of vehicle
      's_min': 0,   # Minimum allowed position
      's_max': 100, # Maximum allowed position
      'v_min': 0,   # Minimum allowed velocity (no driving backwards!)
      'v_max': 35,  # Maximum allowed velocity (speed limit)
      'j_min': -15, # Minimum allowed jerk 
      'j_max': 15,  # Maximum allowed jerk
    }
    
    self.behavior_init_dict = {
      'type': None,           # type of behavior 
      'valid': False,         # is valid or not
      'lane_seq': [],         # target lane seq
      'is_truncated': False,  # can't pass lane seq end or not
      'frenet': None,         # frenet representation of the target lane seq
      'from_to_sum_s': [None, None, None], # from_s, to_s, (to_s - from_s)
      'start_tpoint': None,   # trajectory point of the start state
      'start_fpoint': None,   # frenet point of the start state
      'candidate_paths': [],  # generated list of reference paths
    }
    self.avoid_d_range = avoid_d_range
    self.ref_behavior_params = {
      'lon_s_range': 100.0,
      'lon_s_reso': 5.0,
      'lat_d_reso': 1.0,
    }
    self.ref_behaviors = {}
    self.ispace = InteractionSpace(plan_horizon_T=self.plan_horizon_T,
                                   space_range_x=[-100, 100],
                                   space_range_y=[-100, 100],
                                   space_reso=space_reso,
                                   yaw_reso=(1.0 * TO_RADIAN),
                                   planning_dt=planning_dt,
                                   prediction_dt=prediction_dt)

  '''********************************************************************
                            Behavior Plan
  ********************************************************************'''
  def __behavior_plan(self, nav_map: NavigationMap,
                            time_step: int, 
                            ego_agent: EgoAgent) -> Dict:
    '''
    Get dict of behaviors which follows the format of behavior_init_dict in init()
    :param nav_map: the navigation map, which contains the navigation goal information
    :param time_step: indicates which time step is
    :param ego_agent: store agent states at each time step
    @note the bahavior is represented relative to agent's position along the 'task route'.
    '''
    robot_xyyaw = ego_agent.get_transform(time_step)
    robot_xyyaw = (robot_xyyaw._x, robot_xyyaw._y, robot_xyyaw._yaw)

    # print("\n *********************************************")
    time_point0 = time.time()
    sinfo, _ = nav_map.nearest_lane_and_point('route', robot_xyyaw)

    pred_lanes = nav_map.get_lane_neighbours(sinfo['lane_idx'])['pred']
    print("\n routing from {} with pred_lanes {}".format(sinfo['lane_idx'], pred_lanes))
    if len(pred_lanes) > 0:
      # set routing from the previous lane to reduce frenet calculation error
      sinfo['lane_idx'] = pred_lanes[0]

    route_start_lane_idx = sinfo['lane_idx']
    goal_lane_idx = nav_map.goal['info']['lane_idx']
    
    nxt_goal_time = 3
    while nxt_goal_time > 0:
      nxt_lanes = nav_map.get_lane_neighbours(goal_lane_idx)['succ']
      if len(nxt_lanes) > 0:
        # set routing to the next lane of the original goal lane, to increase the planning distance
        goal_lane_idx = nxt_lanes[0]
      else:
        break
      nxt_goal_time -= 1

    stpoint = TrajectoryPoint(
      timestamp=0.0,
      pos_x=robot_xyyaw[0], pos_y=robot_xyyaw[1], pos_yaw=robot_xyyaw[2], 
      steer=ego_agent.states[time_step]['steering_radian'],
      velocity=ego_agent.states[time_step]['velocity'],
      acceleration=ego_agent.states[time_step]['acceleration']
    )

    ## Init behaviors
    frenet_behaviors = {
      'succ': copy.copy(self.behavior_init_dict),
      # 'left': copy.copy(self.behavior_init_dict), 
      # 'right': copy.copy(self.behavior_init_dict),
    }
    ## types:
    frenet_behaviors['succ']['type'] = 'succ'
    # frenet_behaviors['left']['type'] = 'left'
    # frenet_behaviors['right']['type'] = 'right'

    ## preparations
    lon_s_range = self.ref_behavior_params['lon_s_range']
    lon_s_reso = self.ref_behavior_params['lon_s_reso']
    lat_d_reso = self.ref_behavior_params['lat_d_reso']
    latplanner = LatticePlanner()

    ## hehavior: follow successor lanes
    time_point1 = time.time()
    frenet_behaviors['succ']['valid'], frenet_behaviors['succ']['lane_seq'] = \
      nav_map.init_route(route_start_lane_idx, goal_lane_idx)
    if frenet_behaviors['succ']['valid']:
      content = frenet_behaviors['succ']
      content['is_truncated'], content['frenet'] =\
        nav_map.extract_frenet_path(content['lane_seq'])

      # # behavior: follow left lanes
      # route_start_neighbours = nav_map.get_lane_neighbours(route_start_lane_idx)
      # tmp_goal_neighbours = nav_map.get_lane_neighbours(goal_lane_idx)
      # if (len(route_start_neighbours['left']) > 0) and (len(tmp_goal_neighbours['left']) > 0):
      #   # left-start to left-goal situation.
      #   all_same_direct = nav_map.check_left_right_lane_is_same_direction(
      #     route_start_lane_idx, route_start_neighbours['left'][0], 'left') and\
      #      nav_map.check_left_right_lane_is_same_direction(
      #     goal_lane_idx, tmp_goal_neighbours['left'][0], 'left')
      #   if all_same_direct:
      #     frenet_behaviors['left']['valid'], frenet_behaviors['left']['lane_seq'] =\
      #       nav_map.init_route(route_start_neighbours['left'][0], tmp_goal_neighbours['left'][0])
      # elif (len(route_start_neighbours['left']) > 0):
      #   # left-start to goal situation.
      #   is_same_direct = nav_map.check_left_right_lane_is_same_direction(
      #     route_start_lane_idx, route_start_neighbours['left'][0], 'left')
      #   if is_same_direct:
      #     frenet_behaviors['left']['valid'], frenet_behaviors['left']['lane_seq'] =\
      #       nav_map.init_route(route_start_neighbours['left'][0], goal_lane_idx)

      # # behavior: follow right lanes
      # if (len(route_start_neighbours['right']) > 0) and (len(tmp_goal_neighbours['right']) > 0):
      #   # right-start to right-goal situation.
      #   all_same_direct = nav_map.check_left_right_lane_is_same_direction(
      #     route_start_lane_idx, route_start_neighbours['right'][0], 'right') and\
      #      nav_map.check_left_right_lane_is_same_direction(
      #     goal_lane_idx, tmp_goal_neighbours['right'][0], 'right')
      #   if all_same_direct:
      #     frenet_behaviors['right']['valid'], frenet_behaviors['right']['lane_seq'] =\
      #     nav_map.init_route(route_start_neighbours['right'][0], tmp_goal_neighbours['right'][0])
      # elif (len(route_start_neighbours['right']) > 0):
      #   # right-start to goal situation.
      #   is_same_direct = nav_map.check_left_right_lane_is_same_direction(
      #     route_start_lane_idx, route_start_neighbours['right'][0], 'right')
      #   if is_same_direct:
      #     frenet_behaviors['right']['valid'], frenet_behaviors['right']['lane_seq'] =\
      #       nav_map.init_route(route_start_neighbours['right'][0], goal_lane_idx)

    # print("\n******************************")
    time_point2 = time.time()
    for ckey, content in frenet_behaviors.items():
      lon_max_sample_num = 1
      if (ckey != 'succ'):
        lon_max_sample_num = 1 # all route one candidate

      if content['valid']:
        content['is_truncated'], content['frenet'] =\
          nav_map.extract_frenet_path(content['lane_seq'])
        start_s = content['frenet'].sum_s(robot_xyyaw[0:2])
        max_s = content['frenet'].max_sum_s()

        content['from_to_sum_s'] = [start_s, max_s, (max_s - start_s)]
        content['start_tpoint'] = stpoint
        content['start_fpoint'] = content['frenet'].get_frenet_point(stpoint)

        # print("*****************************************")
        fvalues = content['frenet'].get_frenet_point(stpoint)['frenet']

        # check_xy = content['frenet'].get_cartesian_points(np.array([[start_s, 0.0]]))[0, :]
        # print("debug1 [{:.2f}, {:.2f}]".format(start_s, max_s), 
        #       (robot_xyyaw[0], robot_xyyaw[1]), (check_xy[0], check_xy[1]))
        # print("debug2", fvalues[0])

        # default with robot_d, robot_dd, robot_ddd == 0.0
        d = 0.0
        dd_ds = 0.0
        ddd_ds = 0.0
        content['candidate_paths'] =\
          latplanner.gen_ds_paths(
            content['frenet'],  # frenet path
            fvalues[0:2], [d, dd_ds, ddd_ds], # start state
            [start_s, min(start_s+lon_s_range, max_s)], lon_s_reso, # lon sample
            self.avoid_d_range, lat_d_reso, # lateral sample
            lon_max_sample_num=lon_max_sample_num
          )
        if len(content['candidate_paths']) > 0:
          print("::Behavior[{}] generate paths from".format(ckey),
                "start xyyaw=({:.1f}, {:.1f}, {:.1f});".format(robot_xyyaw[0], robot_xyyaw[1], robot_xyyaw[2]),
                "start s,v,a=[{:.1f}, {:.1f} {:.1f}]; d=[{:.1f}]".format(
                  fvalues[0], fvalues[1], fvalues[2], d))
        # print("gen. {} candidate paths num={}.".format(
        #   ckey, len(content['candidate_paths'])))

    time_point3 = time.time()

    print("Behavior plan calculation times= {:.2f}, {:.2f}, {:.2f};".format(
       time_point1 - time_point0, time_point2 - time_point1, time_point3 - time_point2))

    return frenet_behaviors

  '''********************************************************************
                        Build Interaction Space
  ********************************************************************'''
  def __build_interaction_space(self, time_step: int,
                                      ego_agent: EgoAgent,
                                      agents: List[Dict],
                                      agent_predictions: List,
                                      ignore_rear_agents: bool) -> Dict:
    '''
    Build Interaction Space for analyzing interactions in the search space
    :param time_step: time step of the simulation (utilized in ego_agent)
    :param ego_agent: record ego agent information
    :param agents: list of agent information
    :param agent_predictions: prediction data of agents
    :param ignore_rear_agents: true if ignoring rear agents' prediction data
    '''
    # print(">>>"*10)
    origin = ego_agent.get_transform(time_step)
    origin_xy = [origin._x, origin._y]
    self.ispace.reinit(origin_xy=origin_xy)
    print("__build_interaction_space::debug()")
    # print("  agent amount & predictions amount=", len(agents), len(agent_predictions))

    inv_ori = origin
    inv_ori.inverse()

    # Process prediction kdtrees
    for agent, agent_preds in zip(agents, agent_predictions):
      idx = agent['idx']

      obs_length, obs_width = limit_agent_length_width(length=agent['length'], width=agent['width'])

      xyyaw: XYYawTransform = agent['xyyaw']
      velocity = agent['velocity']
      acceleration = agent['acceleration']

      ego2agent_xyyaw = inv_ori.multiply_from_right(xyyaw)
      rela_angle = math.atan2(ego2agent_xyyaw._y, ego2agent_xyyaw._x)
      if ignore_rear_agents and (math.fabs(rela_angle) > (145 * TO_RADIAN)):
        print("ignore rear agent, idx={}".format(idx))
        continue

      self.ispace.add_predictions(idx, obs_length, obs_width, velocity, acceleration, agent_preds)

  '''********************************************************************
                          S-t Searching
  ********************************************************************'''
  def _plan_given_behaviors(self, nav_map: NavigationMap,
                                  time_step: int, 
                                  ego_agent: EgoAgent,
                                  ref_behaviors: Dict,
                                  agents: List[Dict],
                                  agent_predictions: List,
                                  require_slowdown: bool,
                                  enable_rviz_step: int) -> Dict:
    '''
    Evaluate paths and return the plan results
    :param time_step: time step of the simulation
    :param ego_agent: ego agent
    :param ref_behaviors: planned reference behaviors
    :param agents: agent list
    :param agent_predictions: agent predictions
    :param require_slowdown: forcibly require slowdown
    :param enable_rviz_step: after which step, the rviz is enabled
    :return: dict of planning results, with format {
      'has_result': bool,
      'stva_array': np.ndarray,
      'tva_xyyawcur_array': np.ndarray,
      'speed_limits': np.ndarray
    }
    '''
    ego_state = ego_agent.states[time_step]

    # print(">> _plan_given_behaviors::start()")
    path_cost_sum = 1e-4
    horizon_dist_sum = 1e-4
    action_cost_sum = 1e-4
    reaction_cost_sum = 1e-4

    ti :int = 0 # trajectory index
    result_cache = {}

    # Result collection
    valid_result_num :int = 0
    bi :int = 0 # behavior index
    for bkey, dt in ref_behaviors.items():
      # dt.keys = 'valid', 'lane_seq', 'is_truncated', 'frenet', 
      #           'from_to_sum_s', 'start_tpoint', 'start_fpoint', 
      #           'candidate_paths'
      # @note start_fpoint.s is represented at lane not at behavior/path
      if not dt['valid']:
        continue
      lane_behavior_type = dt['type']
      end_need_stop = dt['is_truncated']
      behavior_s_length = dt['from_to_sum_s'][2]
      # print("behavior_s_length[{}] = {}".format(lane_behavior_type, behavior_s_length))

      candidate_paths = dt['candidate_paths']
      for pi, path_dict in enumerate(candidate_paths):
        # Extract path information, where path_dict keys = 
        #   ['frenet_from_s', 'frenet_to_s', 'frenet_merge_s', 
        #    'offset_d', 'cost_path', 'poly_path', 'path_samples_xyyawcurs']
        poly_path: SplineCurve= path_dict['poly_path']
        path_s_samples: np.ndarray= path_dict['path_samples_s']
        path_s_interval: float= path_dict['path_samples_s_interval']
        path_xyyawcurs: np.ndarray= path_dict['path_samples_xyyawcurs']
        path_cost: float= path_dict['cost_path']

        if math.isnan(path_cost):
          continue # skip this path, polyfit fails

        # Init searcher
        # S-t graph search: use vehicle's speed and accleration as ds, dds 
        #                   as initial state for speed planning 
        start_sva = [path_s_samples[0], ego_state['velocity'], ego_state['acceleration']]

        st_searcher = None
        enable_debug_rviz :bool= (time_step >= enable_rviz_step)
        algo_vars = [
          self.reaction_config['algo_variable1'], 
          self.reaction_config['algo_variable2'],
          self.reaction_config['algo_variable3']
        ]
        reaction_conditions = self.reaction_config['reaction_conditions']

        if enable_debug_rviz:
          # TODO: compare results usign different planenr at certain step
          pass

        if (start_sva[1] < 0.0) or\
           (start_sva[1] > ZoneStvGraphSearch.get_max_v_limit()):
          # skip when initial speed is out of search range
          ColorPrinter.print("yellow", 
            "forcibly set start-v inside speed limits, with {}, {}.".format(
              start_sva, ZoneStvGraphSearch.get_max_v_limit())
          )
          start_sva[1] = max(start_sva[1], 0.0)
          start_sva[1] = min(start_sva[1], (ZoneStvGraphSearch.get_max_v_limit() - 1e-6))

        if self.planner_type == 'none_ca':
          st_searcher = ZoneStvGraphSearch(
            ego_agent, self.ispace, path_s_samples, path_xyyawcurs,
            start_sva=start_sva, search_horizon_T=self.plan_horizon_T,
            planning_dt=self.planning_dt, prediction_dt=self.prediction_dt,
            s_end_need_stop=end_need_stop, path_s_interval=path_s_interval,
            enable_debug_rviz=enable_debug_rviz)
        elif self.planner_type == 'ca':
          st_searcher = ZoneStvGraphSearchCollisionAvoidance(
            ego_agent, self.ispace, path_s_samples, path_xyyawcurs,
            start_sva=start_sva, search_horizon_T=self.plan_horizon_T,
            planning_dt=self.planning_dt, prediction_dt=self.prediction_dt,
            s_end_need_stop=end_need_stop, path_s_interval=path_s_interval,
            enable_debug_rviz=enable_debug_rviz)
        elif self.planner_type == 'ca+':
          st_searcher = ZoneStvGraphSearchCollisionAvoidance(
            ego_agent, self.ispace, path_s_samples, path_xyyawcurs,
            start_sva=start_sva, search_horizon_T=self.plan_horizon_T,
            planning_dt=self.planning_dt, prediction_dt=self.prediction_dt,
            s_end_need_stop=end_need_stop, path_s_interval=path_s_interval,
            enable_debug_rviz=enable_debug_rviz)
        elif (self.planner_type == 'izone_ca') or (self.planner_type == 'izone_ca+'):
          st_searcher = ZoneStvGraphSearchIZoneCollisionAvoidance(
            ego_agent, self.ispace, path_s_samples, path_xyyawcurs,
            start_sva=start_sva, search_horizon_T=self.plan_horizon_T,
            planning_dt=self.planning_dt, prediction_dt=self.prediction_dt,
            s_end_need_stop=end_need_stop, path_s_interval=path_s_interval,
            enable_debug_rviz=enable_debug_rviz)
        elif (self.planner_type == 'izone_rela') or\
             (self.planner_type == 'izone_rela+'):
          st_searcher = ZoneStvGraphSearchIZoneRelations(
            ego_agent, self.ispace, path_s_samples, path_xyyawcurs,
            start_sva=start_sva, search_horizon_T=self.plan_horizon_T,
            planning_dt=self.planning_dt, prediction_dt=self.prediction_dt,
            s_end_need_stop=end_need_stop, path_s_interval=path_s_interval,
            algo_vars=algo_vars, prediction_num=self.predict_traj_mode_num, 
            reaction_conditions=reaction_conditions,
            enable_debug_rviz=enable_debug_rviz)
        else:
          raise ValueError("planner type = {}, which is illegal".format(self.planner_type))

        if require_slowdown:
          st_searcher.set_search_acc_bounds((-1.0, 0.0))

        # Start foward searching
        st_searcher.start_searching()

        # Select best-cost trajectory
        dict_traj = st_searcher.get_planning_results(interval_t=self.planning_dt)
        if dict_traj['has_result']:
          horizon_dist = dict_traj['stva_array'][-1, 0]
          action_cost = dict_traj['sum_action_cost']
          react_cost = dict_traj['sum_react_cost']

          path_cost_sum += path_cost
          horizon_dist_sum += horizon_dist
          action_cost_sum += action_cost
          reaction_cost_sum += react_cost

          result_cache[ti] = {
            'lane_behavior_type': lane_behavior_type,
            'poly_path': poly_path,
            'path_cost': path_cost,
            'horizon_dist': horizon_dist,
            'action_cost': action_cost,
            'reaction_cost': react_cost,
            'stva_array': dict_traj['stva_array'],
            'speed_limits': dict_traj['speed_limits'],
            'edge_counts': dict_traj['edge_counts'],
          }
          valid_result_num += 1

          print("[{}]{} trajectory information=".format(ti, (bi, pi)))
          print(" path_cost={:.2f}; ".format(path_cost))
          print(" horizon_dist={:.2f}; ".format(horizon_dist))
          print(" action_cost={:.2f}; ".format(action_cost))
          print(" react_cost={:.2f}; ".format(react_cost))
          print(" ")
        else:
          stop_stva_array = st_searcher.get_braking_stop_along_the_path(interval_t=self.planning_dt)
          if stop_stva_array.shape[0] > 0:
            result_cache[ti] = {
              'lane_behavior_type': lane_behavior_type,
              'poly_path': poly_path,
              'path_cost': path_cost,
              'horizon_dist': 0.0,
              'action_cost': 1e+3,
              'reaction_cost': 0.0,
              'stva_array': stop_stva_array,
              'speed_limits': np.ones_like(stop_stva_array[:, 1]) * start_sva[1],
              'edge_counts': 0,
            }

            print("[{}]{} trajectory excutes lane-stop".format(ti, (bi, pi)))
            print("stop_stva_array shape=", stop_stva_array.shape)
        
        ti += 1
        st_searcher.visualize_searching()
      bi += 1

    # Pick out the best one
    plan_success = valid_result_num > 0
    best_ti = None
    best_cost = 1e+3
    for ti, content in result_cache.items():
      extra_cost = 0.0
      if content['lane_behavior_type'] != self.last_behavior_type:
        extra_cost = self.change_behavior_cost

      norm_path_cost = content['path_cost'] / path_cost_sum
      norm_horizon_cost = content['horizon_dist'] / horizon_dist_sum
      norm_action_cost = content['action_cost'] / action_cost_sum
      norm_reaction_cost = content['reaction_cost'] / reaction_cost_sum
      get_cost = self.cost_cofficients[0] * norm_path_cost +\
                 self.cost_cofficients[1] * norm_horizon_cost +\
                 self.cost_cofficients[2] * norm_action_cost +\
                 self.cost_cofficients[3] * norm_reaction_cost +\
                 extra_cost
      print(">> plan[{}] with path_cost={:.1f}, horizon_dist={:.1f},".format(
              content['lane_behavior_type'], norm_path_cost, norm_horizon_cost),
            "action_cost={:.1f}, reaction_cost={:.1f}, and keep behavior cost={:.1f};".format(
              norm_action_cost, norm_reaction_cost, extra_cost),
            "weighted sum={:.1f};".format(get_cost)
      )
      if (best_ti == None) or (get_cost < best_cost):
        best_cost = get_cost
        best_ti = ti

    # Transform result to cartesian space
    print("[{}] resulting cmd_traj with cost= {:.2f}.".format(plan_success, best_cost))
    planning_result = {
      'has_result': plan_success,
      'is_already_stop': ego_state['velocity'] < 1e-1,
      'stva_array': None,
      'tva_xyyawcur_array': None,
      'speed_limits': None,
      'edge_counts': 0,
    }
    if len(result_cache.keys()) > 0:
      cache = result_cache[best_ti]
      lane_behavior_type = cache['lane_behavior_type']

      if self.last_behavior_type != lane_behavior_type:
        self.last_behavior_type = lane_behavior_type
        self.change_behavior_cost = self.change_hebavior_init_cost
      else:
        self.change_behavior_cost *= self.change_behavior_cost_decay

      # preparation
      stva_array :np.ndarray= cache['stva_array']
      v_limits :np.ndarray= cache['speed_limits']
      
      poly_path :SplineCurve= cache['poly_path']
      xyyawcurs = poly_path.get_sample_xyyawcur(stva_array[:, 0])

      # fill tva_xyyaw array
      tva_xyyawcur_array = np.zeros([stva_array.shape[0], 7])
      tva_xyyawcur_array[:, :3] = stva_array[:, 1:]
      tva_xyyawcur_array[:, 3:] = xyyawcurs

      # update outputs
      planning_result['has_result'] = plan_success
      planning_result['stva_array'] = stva_array
      planning_result['tva_xyyawcur_array'] = tva_xyyawcur_array
      planning_result['speed_limits'] = v_limits
      planning_result['edge_counts'] = cache['edge_counts']

    return planning_result

  '''********************************************************************
                            Port Functions
  ********************************************************************'''
  def get_ref_behaviors(self) -> Dict:
    '''
    return list of reference behaviors
    '''
    return self.ref_behaviors

  def plan(self, nav_map: NavigationMap,
                 time_step: int, 
                 agents: List[Dict],
                 agent_predictions: List,
                 ego_agent: EgoAgent,
                 require_slowdown: bool,
                 enable_rviz_step: int) -> Tuple:
    '''
    Plan and return planning results
    :param require_slowdown: forcibly require slowdown
    :param enable_rviz_step: after which step, the rviz is enabled
    :return dict of planning result and cost times, whose format follows results of _plan_given_behaviors
    '''
    time_dict = {}
    time_reso :int = 3

    start_s0 = time.time()
    # >> STEP: behavior plan
    start_s = time.time()
    self.ref_behaviors = self.__behavior_plan(nav_map, time_step, ego_agent)
    # print("\nref_behaviors", self.ref_behaviors.keys())
    time_dict['behavior_plan'] = round(time.time() - start_s, time_reso)

    # >> STEP: build interaction space
    flag_ignore_rear :bool=\
      (self.planner_type in self.__planners_ignoring_agent_rears)

    start_s = time.time()
    self.__build_interaction_space(
      time_step=time_step, ego_agent=ego_agent,
      agents=agents, agent_predictions=agent_predictions,
      ignore_rear_agents=flag_ignore_rear)
    time_dict['build_ispace'] = round(time.time() - start_s, time_reso)

    # >> STEP: motion plannings
    start_s = time.time()
    planning_result = self._plan_given_behaviors(
      nav_map, time_step, 
      ego_agent, self.ref_behaviors, 
      agents, agent_predictions,
      require_slowdown=require_slowdown,
      enable_rviz_step=enable_rviz_step)
    time_dict['motion_plan'] = round(time.time() - start_s, time_reso)

    time_dict['total'] = round(time.time() - start_s0, time_reso)
    ColorPrinter.print("green", "Calculation Time: {} (seconds)".format(time_dict))

    return planning_result, time_dict
