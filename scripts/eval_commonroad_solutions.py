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
from envs.commonroad.simulation.utility import limit_agent_length_width

import os
import math 
import sys
from utils.file_io import write_dict2bin, read_dict_from_bin
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from shapely.geometry import Point, LineString, Polygon

from utils.angle_operation import get_normalized_angle
from utils.transform import XYYawTransform

from eval_commonroad_solution import get_polygon, cal_agent_reaction2ego, get_side_linesegs, get_collision_infos

def generate_metrics(scenario_path: str, solution_path: str, record_path: str, metric_save_path: str):
  """refer to the head of this file for the generated metrics

  Args:
      scenario_path (str): absolute path of scenario xml file
      solution_path (str): absolute path of solution xml file
      record_path (str): absolute path of record.bin file
      metric_save_path (str): desried absolution path of metrics bin 
  """
  scenario, planning_problem_set = CommonRoadFileReader(scenario_path).open()
  solution = CommonRoadSolutionReader().open(solution_path)
  record = read_dict_from_bin(record_path, verbose=False)

  if len(solution.planning_problem_solutions) == 0:
    print("Warning, occurs scenario with solution num == 0.")
    return

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
  ego_half_length = get_info['length'] * 0.5
  ego_half_width = get_info['width'] * 0.5

  total_times = 0.0

  ignore_count :int = -1
  collision_dict = {}
  for ii in np.arange(start=t0, stop=(tf+1), step=1):
    ego_state_ii = ego.state_at_time_step(ii)

    ego_poly = get_polygon(
      ego_state_ii.position, ego_state_ii.orientation, ego_half_length, ego_half_width,
      rear_ignore=0.25)
    seg_tags, norm_list, ego_segs = get_side_linesegs(
      ego_state_ii.position, ego_state_ii.orientation, ego_half_length, ego_half_width)

    # calculate speed, acc and jerk
    v_list.append(ego_state_ii.velocity)

    if (ii != tf):
      a = (ego.state_at_time_step(ii+1).velocity - ego_state_ii.velocity)/dt
      a_list.append(a)
      if (ii != 0):
        get_jerk = (a - a_list[ii-1])/dt
        j_list.append(max(-30.0, min(get_jerk, 30.0)))

    # calculate clearance
    distance_list = []
    for obs in agents:
      # check clearance
      if obs.state_at_time(ii) is not None:
        obs_pos = obs.state_at_time(ii).position
        obs_ori = obs.state_at_time(ii).orientation
        obs_v = obs.state_at_time(ii).velocity

        obs_length, obs_width = limit_agent_length_width(
          obs.obstacle_shape.length, obs.obstacle_shape.width)
        half_width = obs_width * 0.5
        half_length = obs_length * 0.5

        obs_poly = get_polygon(obs_pos, obs_ori, half_length, half_width)

        get_clear = ego_poly.distance(obs_poly)
        distance_list.append(get_clear)

        if get_clear < 1e-3:
          _, max_overlap_length, collision_type = get_collision_infos(
            ego_state_ii.velocity, ego_segs, norm_list, seg_tags, obs_poly, obs_v)

          # if max_overlap_length > 0.1: # ignore sidewipe
          if not obs.obstacle_id in collision_dict.keys():
            collision_dict[obs.obstacle_id] = [ii, 1, 
              {'stop_ego_collision': 0, 'stop_track_collision': 0, 
               'front': 0, 'side': 0, 'rear': 0}] # time_stamp record, collision times
            collision_dict[obs.obstacle_id][2][collision_type] = 1
          
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

  reaction_dcc_efforts = []
  for obs_id, obs in enumerate(agents):
    acc_list = cal_agent_reaction2ego(ego, obs, step_dt=dt)
    if len(acc_list) > 0:
      reaction_dcc_efforts = reaction_dcc_efforts + (np.square(acc_list) * dt).tolist()

  metrics['reaction_dcc_efforts'] = np.array(reaction_dcc_efforts)

  # save metrics to the metrics_path
  # print("with collision_times=", collision_times)
  # print("metric_save_path=", metric_save_path)
  write_dict2bin(metrics, metric_save_path, verbose=False)

import envs.config
import conf.__init__
import yaml
from utils.file_io import extract_folder_file_list

if __name__ == '__main__':
  cfg = None
  with open(os.path.join(os.path.dirname(conf.__init__.__file__), 
            'eval_tags.yaml')) as config_file:
    cfg = yaml.safe_load(config_file)['eval_config']
  tag_str_list = cfg['eval_tags'] # get eval solution tag list

  # TODO: add stop rate of all traffic agents in the simulation (at the end several frames)

  for eval_tag in tag_str_list:
    root_dir = envs.config.get_dataset_exp_folder('commonroad', 'exp_plan')
    result_scenarios_dir = os.path.join(root_dir, '{}/result_scenarios'.format(eval_tag))
    solutions_dir = os.path.join(root_dir, '{}/solutions'.format(eval_tag))
    metrics_dir = envs.config.get_root2folder(root_dir, '{}/evals'.format(eval_tag))

    scene_dir_file_list = extract_folder_file_list(result_scenarios_dir)
    solu_dir_file_list = extract_folder_file_list(solutions_dir)

    scenarios = sorted([fname for fname in scene_dir_file_list if '.xml' in fname], key=lambda p: int(p.split('[')[1].split(']')[0]))
    solutions = sorted([fname for fname in solu_dir_file_list if '.xml' in fname], key=lambda p: int(p.split('[')[1].split(']')[0]))
    records = sorted([fname for fname in solu_dir_file_list if '.bin' in fname], key=lambda p: int(p.split('[')[1].split(']')[0]))

    print("Evaluate solution with tag={}.".format(eval_tag))
    assert len(scenarios) == len(solutions),\
      "Error1, num not equal, need to delete the whole folder and re-run once {}, {}.".format(
        len(scenarios), len(solutions))
    assert len(scenarios) == len(records),\
      "Error2, num not equal, need to delete the whole folder and re-run once {}, {}.".format(
        len(scenarios), len(records))

    jj :int= 0
    eval_num :int= 0
    skip_num :int= 0
    city_set = set()
    location_set = set()
    total_num :int= len(scenarios)
    for scenario_fname, solu_fname, record_fname in zip(scenarios, solutions, records):
      scenario_path = os.path.join(result_scenarios_dir, scenario_fname)
      solution_path = os.path.join(solutions_dir, solu_fname)
      record_path = os.path.join(solutions_dir, record_fname)
      metric_path = os.path.join(metrics_dir, '[{}]metric.bin'.format(jj))
      jj += 1

      print("\rProcessing scenario {}/{}.".format(jj, total_num), end="")
    
      # city_name [0]_DEU_A9-1_1_I-1-1_results.xml
      location_name = scenario_path.split('/')[-1].split('-')[0].split(']_')[1]
      city_name = location_name.split('_')[0]

      city_set.add(city_name)
      location_set.add(location_name)
      # try:
      generate_metrics(scenario_path, solution_path, record_path, metric_path)
      eval_num += 1
      # except Exception as einfo:
      #   skip_num += 1
      #   print("Skip scenrio={}", scenario_path, 
      #     "\nBecause {} (some specific map traffic light format is unrecognized)",format(einfo))

      # print("\n test plot_path")
      # plot_path(scenario_path, solution_path, metric_path)

    print("eval solutions from folder:")
    print(" scenarios folder=", result_scenarios_dir)
    print(" solutions folder=", solutions_dir)
    print(" saved to metrics folder=", metrics_dir)
    print("Final with {} cities processed, {} locations processed,".format(len(city_set), len(location_set)))
    print(" {} scenarios processed, and {} scenarios skipped".format(eval_num, skip_num))
    print(" ")
