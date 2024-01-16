# Cost evaluations include
#   1. accleration cost= /sum_{t_0}^{t_f}a^2 dt
#   2. jerk cost= /sum_{t_0}^{t_f}j^2 dt
#   3. distance to obstacles= /sum_{t_0}^{t_f} max(\delta_1, \delta_2, ..., \delta_o) dt
#      where, \delta = e ^ {-d_o}, d_o is the distance of ego vehicle to an obstacle
#   4. path length= /sum_{t_0}^{t_f} v dt
#   5. reaction cost of other agents= /sum_{t_0}^{t_1}dcc^2 dt
#      where [t0, t1] is the time window when agents are reacting to agents.

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution import CommonRoadSolutionReader

import os
import math 
import sys
from utils.file_io import write_dict2bin, read_dict_from_bin
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from shapely.geometry import Point, LineString, Polygon

from envs.commonroad.simulation.utility import limit_agent_length_width
from utils.angle_operation import get_normalized_angle
from utils.transform import XYYawTransform

TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE

# TODO: plot_path() not more support to visualize reaction_loss values
# def plot_path(scenario_path: str, solution_path:str, metrics_path: str, agents_ids: list=[], plot_ego: bool=True, plot_all_agents: bool=True, plot_metrics: bool=True):
#   """_summary_

#   Args:
#       scenario_path (str): absolute path of scenario xml file
#       agents_ids (list, optional): relative id of agents list. Defaults to [].
#       plot_ego (bool, optional): if plot ego path. Defaults to True.
#       plot_all_agents (bool, optional): if yes, agents_id is not considered. Defaults to True.
#   """
#   if (not plot_ego and not plot_all_agents and len(agents)==0):
#       print("nothing to plot")
#       return
#   scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
#   solution = CommonRoadSolutionReader().open(solution_path)
#   ego = solution.planning_problem_solutions[0].trajectory
#   agents = scenario.dynamic_obstacles
#   dt = scenario.dt
#   if not plot_all_agents:
#       agents = [agents[id] for id in agents_ids]
#   else:
#       agents_ids = [i for i in range(len(agents))]
#   t0 = ego.initial_time_step
#   tf = ego.final_state.time_step
#   ego_path = [ego.state_at_time_step(t).position for t in range(t0, (tf+1), 1)]
#   agents_path = []
#   for obs in agents:
#       obs_path = [obs.state_at_time(t).position for t in range(t0, (tf+1), 1) if obs.state_at_time(t) is not None]
#       agents_path.append(obs_path)
#   metrics = read_dict_from_bin(metrics_path)
#   reaction_loss = metrics['reaction_loss']
#   if not plot_all_agents:
#       reaction_loss = [reaction_loss[id] for id in agents_ids]
#   reaction_loss = [reaction_loss[i] for i in range(len(reaction_loss)) if reaction_loss[i]['ref_tf'] is not None]
#   figure_num = len(reaction_loss)

#   if figure_num == 0 or not plot_metrics:      
#     figure = plt.figure()
#     if plot_ego:
#       ego_x = [pos[0] for pos in ego_path]
#       ego_y = [pos[1] for pos in ego_path]
#       plt.plot(ego_x, ego_y, label="ego path", marker='.', markevery=[0, 25, 55])
#     obs_index = 0
#     for obs in agents_path:
#       obs_x = [pos[0] for pos in obs]
#       obs_y = [pos[1] for pos in obs]
#       if (obs_index == 1):
#           plt.plot(obs_x, obs_y, label=f'agent {agents_ids[obs_index]}', marker='.', markevery=[0, 25, 55])
#       else:
#           plt.plot(obs_x, obs_y, label=f'agent {agents_ids[obs_index]}', marker='.', markevery=[0])
      
#       obs_index += 1
#   else:
#     for f_ind in range(figure_num):
#       t0 = reaction_loss[f_ind]['t0']
#       tf = reaction_loss[f_ind]['tf']
#       agent_id = reaction_loss[f_ind]['agent_relative_id']
#       agent = agents[agent_id]
#       v_list = [agent.state_at_time(i).velocity for i in range(t0, (tf+1), 1)]
#       acc_list = [(agent.state_at_time(i).velocity-agent.state_at_time(i-1).velocity)/dt for i in range(t0, tf+1, 1)]
#       # plot path with intersection point
#       plt.subplot(2,1,2)
#       if plot_ego:
#           ego_x = [pos[0] for pos in ego_path]
#           ego_y = [pos[1] for pos in ego_path]
#           plt.plot(ego_x, ego_y, label="ego path", marker='.', markevery=[0, t0, tf])
#       obs_index = 0
#       for obs in agents_path:
#           obs_x = [pos[0] for pos in obs]
#           obs_y = [pos[1] for pos in obs]
#           if (agents_ids[obs_index] == agent_id):
#               plt.plot(obs_x, obs_y, label=f'agent {agents_ids[obs_index]}', marker='.', markevery=[0, t0, tf])
#           else:
#               plt.plot(obs_x, obs_y, label=f'agent {agents_ids[obs_index]}', marker='.', markevery=[0])
#           obs_index += 1
#       plt.legend(loc='center right', bbox_to_anchor=(1.05, 0.5))
#       # plot velocity of interacted ego and agent
#       plt.subplot(2, 2, 1)
#       t = np.arange(t0, tf+1, 1)
#       plt.plot(t, v_list)  
#       ego_v_list = [ego.state_at_time_step(i).velocity for i in range(t0, tf+1, 1)]
#       ego_dcc_list = [ego.state_at_time_step(i).velocity-ego.state_at_time_step(i-1).velocity for i in range(t0, tf+1, 1)]
#       plt.plot(t, ego_v_list, label='velocity of ego')
#       plt.plot(t, v_list, label=f'velocity of agent {agent_id}')
#       plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25))
#       # plot dcc of interacted ego and agent
#       plt.subplot(2, 2, 2)
#       plt.plot(t, ego_dcc_list, label='dcc of ego')
#       plt.plot(t, acc_list, label=f'dcc of agent {agent_id}')
#       plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25))
#   plt.show()

def cal_agent_reaction2ego(ego, agent, relative_x_condition: float = -8.0,
    distance_condition: float = 40.0, radian_condition: float = (30 * TO_RADIAN),
    step_dt: float = 0.1) -> List:
  """search if an ego and an agent have the interaction trend. If it is, give the intersection time t0 and tf.

  Args:
      ego (solution.planning_problem_solutions[0].trajectory): contain ego state information
      agent (scenario.dynamic_obstacles element): contain agent information
      relative_x_condition (float): relative distance condition along x axis to consider the reaction function
      distance_condition (float): distance condition to consider the reaction function
      radian_condition (float): angle condition to consider the reaction function
      step_dt (float): delta t of the time step
  Returns:
      List: return list of deceleration of surrounding agents
  """
  t0 = ego.initial_time_step
  tf = ego.final_state.time_step
  min_dis = np.inf
  overlap_t_index = np.inf
  ego_t_index = np.inf

  record_dccs = []
  for ii in range(t0, (tf+1), 1):
    ego_state = ego.state_at_time_step(ii)
    agent_state = agent.state_at_time(ii)
    if agent_state is not None:
      ego_xyyaw = XYYawTransform(x=ego_state.position[0], 
        y=ego_state.position[1], yaw_radian=ego_state.orientation)
      inv_ego_xyyaw = XYYawTransform(x=ego_state.position[0], 
        y=ego_state.position[1], yaw_radian=ego_state.orientation)
      inv_ego_xyyaw.inverse()

      agent_xyyaw = XYYawTransform(x=agent_state.position[0], 
        y=agent_state.position[1], yaw_radian=agent_state.orientation)
      inv_agent_xyyaw = XYYawTransform(x=agent_state.position[0], 
        y=agent_state.position[1], yaw_radian=agent_state.orientation)
      inv_agent_xyyaw.inverse()

      ego2agent_xyyaw = inv_ego_xyyaw.multiply_from_right(agent_xyyaw)
      agent2ego_xyyaw = inv_agent_xyyaw.multiply_from_right(ego_xyyaw)

      abs_agent2ego_yaw = math.fabs(agent2ego_xyyaw._yaw)

      # ego2agent_angle_range = 150.0*TO_RADIAN
      abs_ego2agent_yaw = math.fabs(
        math.atan2(ego2agent_xyyaw._y, ego2agent_xyyaw._x))

      get_dist = math.dist(ego_state.position, agent_state.position)
      if (get_dist < distance_condition) and\
         (agent2ego_xyyaw._x > relative_x_condition) and\
         (abs_agent2ego_yaw < radian_condition): # and (abs_ego2agent_yaw >= ego2agent_angle_range):
        # Calculate the agent's deceleration when meeting
        #   cond1: distance range is within distance_condition
        #   cond2+3: ego is front of agent
        #   cond4: agent is lateral to ego
        pii = ii - 1
        pstate = agent.state_at_time(pii)

        nii = ii + 1
        nstate = agent.state_at_time(nii)
        acc = None
        if pstate:
          acc = (agent_state.velocity - pstate.velocity) / step_dt
        elif nstate:
          acc = (nstate.velocity - agent_state.velocity) / step_dt

        if (acc is not None) and (-10.0 <= acc) and (acc < 0.0):
          record_dccs.append(acc)

  return record_dccs

def get_side_linesegs(center_xy: Tuple, orientation: float, half_length: float, half_width: float) -> Tuple:
  '''
  Return robot's list of line segment and their belonging seg_tag
  As shown in the below, [f]: front, [s]: side, [r]: rear
    0 f f f 5 
    s       s
    s       s
    s       s
    s       s
    s       s
    1       4
    r       r
    2 r r r 3,
  where frontrear_coef determines how much length of the side are deemed as part of from/rear
  '''
  frontrear_coef = 0.7
  part_len = half_length * frontrear_coef
  _leftrear_len = half_length * (1.0 - frontrear_coef)
  _side_len = part_len * 2.0

  check_points = [
    # from left side points: from top to bottom
    [half_length, half_width], [-part_len, half_width], [-half_length, half_width],
    # from right side points: from bottom to top
    [-half_length, -half_width], [-part_len, -half_width], [half_length, -half_width]
  ]

  # transform from local frame to global frame
  sin = math.sin(orientation)
  cos = math.cos(orientation)
  xy_points = []
  for offset_xy in check_points:
    dx, dy = offset_xy
    # xyyaw._x = relative_x * self._cos_yaw - relative_y * self._sin_yaw + self._x
    # xyyaw._y = relative_x * self._sin_yaw + relative_y * self._cos_yaw + self._y
    x = dx * cos - dy * sin + center_xy[0]
    y = dx * sin + dy * cos + center_xy[1]
    xy_points.append([x, y])

  # Init line segments
  front_seg = LineString([xy_points[5], xy_points[0]])

  left_seg = LineString([xy_points[0], xy_points[1]])
  right_seg = LineString([xy_points[4], xy_points[5]])
  
  leftrear_seg = LineString([xy_points[1], xy_points[2]])
  rear_seg = LineString([xy_points[2], xy_points[3]])
  rightrear_seg = LineString([xy_points[3], xy_points[4]])

  tag_list = ['front', 'side', 'side', 'rear', 'rear', 'rear']
  norm_list = [half_width, 
               _side_len, _side_len, 
               _leftrear_len, half_width, _leftrear_len]
  seg_list = [front_seg, 
              left_seg, right_seg, 
              leftrear_seg, rear_seg, rightrear_seg]
  return tag_list, norm_list, seg_list

def get_collision_infos(ego_speed: float, ego_segs: List, norm_length: List, seg_tags, obs_poly: Polygon, obs_speed: float) -> Tuple:
  '''
  Return collision informations
  '''
  intesected_lengths = [seg.intersection(obs_poly).length for seg in ego_segs]
  norm_lengths = (np.array(intesected_lengths) / np.array(norm_length)).tolist()

  index = (norm_lengths.index(max(norm_lengths)))
  collided_seg = seg_tags[index]
  max_overlap_length = norm_lengths[index]

  ##################################################################################################
  # NUPLAN-Collision metrics code: nuplan.find_new_collisions()
  # four types of collisions
  # 1. STOPPED_EGO_COLLISION
  # 2. STOPPED_TRACK_COLLISION
  # 3. ACTIVE_FRONT_COLLISION
  # 4. ACTIVE_LATERAL_COLLISION
  #
  # # Collisions at (close-to) zero ego speed
  #   if is_ego_stopped:
  #       collision_type = CollisionType.STOPPED_EGO_COLLISION
  #   # Collisions at (close-to) zero track speed
  #   elif is_track_stopped(tracked_object):
  #       collision_type = CollisionType.STOPPED_TRACK_COLLISION
  #   # Rear collision when both ego and track are not stopped
  #   elif is_agent_behind(ego_state.rear_axle, tracked_object.box.center):
  #       collision_type = CollisionType.ACTIVE_REAR_COLLISION
  #   # Front bumper collision when both ego and track are not stopped
  #   elif LineString(
  #       [
  #           ego_state.car_footprint.oriented_box.geometry.exterior.coords[0],
  #           ego_state.car_footprint.oriented_box.geometry.exterior.coords[3],
  #       ]
  #   ).intersects(tracked_object.box.geometry):
  #       collision_type = CollisionType.ACTIVE_FRONT_COLLISION
  #   # Lateral collision when both ego and track are not stopped
  #   else:
  #       collision_type = CollisionType.ACTIVE_LATERAL_COLLISION
  ##################################################################################################
  # TOW TYPES OF COLLISION ARE CONSIDERED in nuplan.classify_at_fault_collisions()
  # : ACTIVE_FRONT_COLLISION + STOPPED_TRACK_COLLISION + ACTIVE_LATERAL_COLLISION(when lane change)
  #
  # Add front collisions and collisions with stopped track to at fault collisions
  # collisions_at_stopped_track_or_active_front = collision_data.collision_type in [
  #     CollisionType.ACTIVE_FRONT_COLLISION,
  #     CollisionType.STOPPED_TRACK_COLLISION,
  # ]
  # # Add lateral collisions if ego was in multiple lanes (e.g. during lane change) to at fault collisions
  # collision_at_lateral = collision_data.collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
  # if collisions_at_stopped_track_or_active_front or (
  #     ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
  # )

  cond_stop_v = 0.1
  get_collision_type = None
  if ego_speed <= cond_stop_v:
    if collided_seg == 'rear':
      get_collision_type = 'rear' # count rear hits even when is at stop
    else:
      get_collision_type = 'stop_ego_collision'
  elif obs_speed <= cond_stop_v:
    get_collision_type = 'stop_track_collision'
  else:
    get_collision_type = collided_seg # 'front', 'rear', 'side'

  return collided_seg, max_overlap_length, get_collision_type

def get_polygon(center_xy: Tuple, orientation: float, 
                half_length: float, half_width: float, rear_ignore: float= 0.0) -> Polygon:
  '''
  :param rear_ignore: ignore certain length of the polygon's rear part 
    as agents in commonroad are not so critical about the shape.
  Return polygon
  '''
  # xyyaw._x = relative_x * self._cos_yaw - relative_y * self._sin_yaw + self._x
  # xyyaw._y = relative_x * self._sin_yaw + relative_y * self._cos_yaw + self._y
  sin = math.sin(orientation)
  cos = math.cos(orientation)
  xy_points = []
  front_length = half_length
  rear_length = -half_length + rear_ignore
  for offset_xy in [[front_length, half_width], [rear_length, half_width], 
                    [rear_length, -half_width], [front_length, -half_width]]:
    dx, dy = offset_xy
    x = dx * cos - dy * sin + center_xy[0]
    y = dx * sin + dy * cos + center_xy[1]
    xy_points.append([x, y])
  
  return Polygon(tuple(xy_points))

def cal_print_metrics(scenario_fname: str, scenario_path: str, 
    solution_path: str, record_path: str) -> int:
  """refer to the head of this file for the generated metrics

  Args:
      scenario_fname (str): scenario file name
      scenario_path (str): absolute path of scenario xml file
      solution_path (str): absolute path of solution xml file
      record_path (str): absolute path of record.bin file
  """
  scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
  solution = CommonRoadSolutionReader().open(solution_path)
  record = read_dict_from_bin(record_path, verbose=False)

  if len(solution.planning_problem_solutions) == 0:
    print("Warning, occurs scenario with solution num == 0.")
    return 0

  ego = solution.planning_problem_solutions[0].trajectory
  agents = scenario.dynamic_obstacles
  t0 = ego.initial_time_step
  tf = ego.final_state.time_step
  dt :float= scenario.dt
  
  metrics = {}
  # ego (speed variation): (v[t+1]-v[t])/dt
  a_list = []
  # ego (speed): v[t]
  v_list = []
  # ego (jerk): (a_list[t+1]-a_list[t])/dt
  j_list = []
  # Distance to obstacles: minimum distance of front obstacle to ego
  clearance_list = []

  get_info = record['ego_agent_info']
  ego_length, ego_width = get_info['length'], get_info['width']
  ego_half_length = ego_length * 0.5
  ego_half_width = ego_width * 0.5

  total_times = 0.0

  ignore_count :int = -1
  collision_dict = {}
  side_dist_list = []
  for ii in np.arange(start=t0, stop=(tf+1), step=1):
    ego_state_ii = ego.state_at_time_step(ii)

    ego_poly = get_polygon(
      ego_state_ii.position, ego_state_ii.orientation, ego_half_length, ego_half_width, rear_ignore=0.25)
    seg_tags, norm_list, ego_segs = get_side_linesegs(
      ego_state_ii.position, ego_state_ii.orientation, ego_half_length, ego_half_width)

    # calculate speed, acc and jerk
    v_list.append(ego_state_ii.velocity)

    if (ii != tf):
      a = (ego.state_at_time_step(ii+1).velocity - ego_state_ii.velocity)/dt
      a_list.append(a)
      if (ii != 0):
        get_jerk = (a - a_list[ii-1])/dt
        # if math.fabs(get_jerk) >= 30.0:
        #   print("get_jerk=", a, a_list[ii-1], get_jerk)
        #   print(" ")
        j_list.append(max(-30.0, min(get_jerk, 30.0)))

    # calculate clearance
    distance_list = []
    for obs in agents:
      # check clearance
      if obs.state_at_time(ii) is not None:
        obs_pos = obs.state_at_time(ii).position
        obs_ori = obs.state_at_time(ii).orientation
        obs_v = obs.state_at_time(ii).velocity

        obs_length, obs_width = limit_agent_length_width(obs.obstacle_shape.length, obs.obstacle_shape.width)

        half_width = obs_width * 0.5
        half_length = obs_length * 0.5
        obs_poly = get_polygon(obs_pos, obs_ori, half_length, half_width)

        get_clear = ego_poly.distance(obs_poly)
        distance_list.append(get_clear)

        if get_clear < 1e-3:
          _, max_overlap_length, collision_type = get_collision_infos(
            ego_state_ii.velocity, ego_segs, norm_list, seg_tags, obs_poly, obs_v)

          if not obs.obstacle_id in collision_dict.keys():
            # time_stamp record, collision times, {collision_times at each pattern}
            collision_dict[obs.obstacle_id] = [ii, 1, 
              {'stop_ego_collision': 0, 'stop_track_collision': 0, 
               'front': 0, 'side': 0, 'rear': 0}] # time_stamp record, collision times
            collision_dict[obs.obstacle_id][2][collision_type] = 1

          side_dist_list.append(
            {'timestep': ii, 'lw': [obs_length, obs_width], 
              collision_type: max_overlap_length}
          )
          # print({'timestep': ii, 'lw': [obs_length, obs_width], 
          #        collided_seg: max_overlap_length})
          
          # add one collision time when collision time stamp is not continuous
          if ii - collision_dict[obs.obstacle_id][0] >= 10: # skip 10 time steps
            collision_dict[obs.obstacle_id][1] += 1
          collision_dict[obs.obstacle_id][0] = ii

    if (len(distance_list) != 0):
      min_dist = min(distance_list)
      clearance_list.append(min_dist)
    
    total_times += 1.0

  collision_times = {
    'stop_ego_collision': 0, 
    'stop_track_collision': 0,
    'front': 0, 'side': 0, 'rear': 0
  }
  for _, content in collision_dict.items():
    collision_times['stop_ego_collision'] += content[1] * content[2]['stop_ego_collision']
    collision_times['stop_track_collision'] += content[1] * content[2]['stop_track_collision']
    collision_times['front'] += content[1] * content[2]['front']
    collision_times['side'] += content[1] * content[2]['side']
    collision_times['rear'] += content[1] * content[2]['rear']

  v_array = np.array(v_list)
  acc_array = np.array(a_list)
  jerk_array = np.array(j_list)
  clearance_array = np.array(clearance_list) # may have negative

  metrics['v_data'] = v_array
  metrics['acc_data'] = acc_array
  metrics['jerk_data'] = jerk_array
  metrics['clearance_data'] = clearance_array

  metrics['acc_loss'] = np.mean(np.square(acc_array) * dt)
  metrics['jerk_loss'] = np.mean(np.square(jerk_array)* dt) 
  metrics['collision_times'] = collision_times
  metrics['path_length'] = np.sum(v_array* dt)

  reaction_loss = []
  for obs_id, obs in enumerate(agents):
    dcc_list = cal_agent_reaction2ego(ego, obs, step_dt=dt)
    if len(dcc_list) > 0:
      loss_array = np.square(np.array(dcc_list)) * dt
      reaction_loss = reaction_loss + loss_array.tolist()

  metrics['reaction_loss'] = np.array(reaction_loss)

  legal_collision_times =\
    metrics['collision_times']['front'] + metrics['collision_times']['side'] + metrics['collision_times']['stop_track_collision']
  rear_times = metrics['collision_times']['rear'] # + metrics['collision_times']['stop_ego_collision']
  if (legal_collision_times > 0) or (rear_times > 0):
    print("Processing scenario= {}.".format(scenario_fname))
    # print('path length', round(metrics['path_length'], 1))
    # print('side_dist_list', side_dist_list)
    print('collision_times=', metrics['collision_times'])

  return legal_collision_times

import envs.config
import conf.__init__
import yaml
from utils.file_io import extract_folder_file_list

if __name__ == '__main__':
  cfg = None
  with open(os.path.join(os.path.dirname(conf.__init__.__file__), 
            'eval_tag.yaml')) as config_file:
    cfg = yaml.safe_load(config_file)['eval_config']

  eval_tag :str= cfg['eval_tag']

  eval_scene_key_str :str= None
  # eval_scene_key_str :str= 'BEL_Putte-16_5_I-1-1' # 760
  # eval_scene_key_str :str= 'BEL_Putte-17_4_I-1-1' # 765
  # eval_scene_key_str :str= 'BEL_Zwevegem-2_1_I-1-1' # 836
  # eval_scene_key_str :str= '800_BEL_Wervik-1_4_I-1-1'

  root_dir = envs.config.get_dataset_exp_folder('commonroad', 'exp_plan')
  result_scenarios_dir = os.path.join(root_dir, '{}/result_scenarios'.format(eval_tag))
  solutions_dir = os.path.join(root_dir, '{}/solutions'.format(eval_tag))
  metrics_dir = envs.config.get_root2folder(root_dir, '{}/evals'.format(eval_tag))

  scene_dir_file_list = extract_folder_file_list(result_scenarios_dir)
  solu_dir_file_list = extract_folder_file_list(solutions_dir)

  scenarios = sorted([fname for fname in scene_dir_file_list if '.xml' in fname], key=lambda p: int(p.split('[')[1].split(']')[0]))
  solutions = sorted([fname for fname in solu_dir_file_list if '.xml' in fname], key=lambda p: int(p.split('[')[1].split(']')[0]))
  records = sorted([fname for fname in solu_dir_file_list if '.bin' in fname], key=lambda p: int(p.split('[')[1].split(']')[0]))

  print("//"*35)

  total_collision_times :int= 0
  total_num :int= len(scenarios)
  for scenario_fname, solu_fname, record_fname in zip(scenarios, solutions, records):
    enable_eval = (eval_scene_key_str == None) or (eval_scene_key_str in scenario_fname)
    
    if enable_eval:
      scenario_path = os.path.join(result_scenarios_dir, scenario_fname)
      solution_path = os.path.join(solutions_dir, solu_fname)
      record_path = os.path.join(solutions_dir, record_fname)
    
      # city_name [0]_DEU_A9-1_1_I-1-1_results.xml
      location_name = scenario_path.split('/')[-1].split('-')[0].split(']_')[1]
      city_name = location_name.split('_')[0]

      total_collision_times += cal_print_metrics(
        scenario_fname, scenario_path, solution_path, record_path)
  
  print("\nEvaluate solution with tag={}.".format(eval_tag))
  print("Final with total_collision_times=", total_collision_times)