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

from eval_commonroad_solution import get_polygon, cal_agent_reaction2ego, get_side_linesegs, get_collision_infos

def cal_metrics(scenario_fname: str, scenario_path: str, 
    solution_path: str, record_path: str) -> Tuple:
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
    return None, None

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
  if get_info == None:
    return None, None
  ego_length, ego_width = get_info['length'], get_info['width']
  ego_half_length = ego_length * 0.5
  ego_half_width = ego_width * 0.5

  total_times = 0.0

  collision_dict = {}
  side_dist_list = []
  for ii in np.arange(start=t0, stop=(tf+1), step=1):
    ego_state_ii = ego.state_at_time_step(ii)

    ego_poly = get_polygon(
      ego_state_ii.position, ego_state_ii.orientation, ego_half_length, ego_half_width)
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

        obs_length, obs_width = limit_agent_length_width(obs.obstacle_shape.length, obs.obstacle_shape.width)

        half_width = obs_width * 0.5
        half_length = obs_length * 0.5
        obs_poly = get_polygon(obs_pos, obs_ori, half_length, half_width)

        get_clear = ego_poly.distance(obs_poly)
        distance_list.append(get_clear)

        if get_clear < 1e-3:
          collided_seg, max_overlap_length, _ = get_collision_infos(
            ego_state_ii.velocity, ego_segs, norm_list, seg_tags, obs_poly, obs_v)

          if not obs.obstacle_id in collision_dict.keys():
            collision_dict[obs.obstacle_id] = [ii, 1, 
              {'front': 0, 'side': 0, 'rear': 0}] # time_stamp record, collision times
            collision_dict[obs.obstacle_id][2][collided_seg] = 1

          side_dist_list.append(
            {'timestep': ii, 'lw': [obs_length, obs_width], 
              collided_seg: max_overlap_length}
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
    'front': 0, 'side': 0, 'rear': 0
  }
  for _, content in collision_dict.items():
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

  front_side_times = metrics['collision_times']['front'] + metrics['collision_times']['side']
  rear_times = metrics['collision_times']['rear']

  metrics['fail_rate'] = 1.0 - (record['plan_success_steps'] / max(1.0, record['total_simu_steps']))
  return front_side_times, metrics

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

  root_dir = envs.config.get_dataset_exp_folder('commonroad', 'exp_plan')
  result_scenarios_dir = os.path.join(root_dir, '{}/result_scenarios'.format(eval_tag))
  solutions_dir = os.path.join(root_dir, '{}/solutions'.format(eval_tag))
  metrics_dir = os.path.join(root_dir, '{}/evals'.format(eval_tag))
  videos_dir = os.path.join(root_dir, '{}/videos'.format(eval_tag))

  scene_dir_file_list = extract_folder_file_list(result_scenarios_dir)
  solu_dir_file_list = extract_folder_file_list(solutions_dir)
  video_dir_file_list = extract_folder_file_list(videos_dir)

  scenarios = sorted([fname for fname in scene_dir_file_list if '.xml' in fname], key=lambda p: int(p.split('[')[1].split(']')[0]))
  solutions = sorted([fname for fname in solu_dir_file_list if '.xml' in fname], key=lambda p: int(p.split('[')[1].split(']')[0]))
  records = sorted([fname for fname in solu_dir_file_list if '.bin' in fname], key=lambda p: int(p.split('[')[1].split(']')[0]))

  print("//"*35)
  get_list = []

  total_collision_times :int= 0
  assert len(scenarios) == len(solutions), "Error1, num not equal, need to delete the whole folder to rerun once {}, {}.".format(
    len(scenarios), len(solutions))
  assert len(scenarios) == len(records), "Error2, num not equal, need to delete the whole folder to rerun once {}, {}.".format(
    len(scenarios), len(records))
  total_num :int= len(scenarios)
  ii :int= 0
  skip_scene = []
  for scenario_fname, solu_fname, record_fname in zip(scenarios, solutions, records):    
    print("\rprocessing {}/{}".format(ii, total_num), end="")
    ii += 1

    scenario_path = os.path.join(result_scenarios_dir, scenario_fname)
    solution_path = os.path.join(solutions_dir, solu_fname)
    record_path = os.path.join(solutions_dir, record_fname)
  
    # city_name [0]_DEU_A9-1_1_I-1-1_results.xml
    location_name = scenario_path.split('/')[-1].split('-')[0].split(']_')[1]
    city_name = location_name.split('_')[0]

    sname = scenario_fname.split(']_')[1].split('_results.xml')[0]
    flags = np.array([sname in vname for vname in video_dir_file_list])
    vname = video_dir_file_list[np.where(flags)[0][0]]
    scenario_index = int(vname.split('_')[0])

    collision_times, metrics = cal_metrics(
      scenario_fname, scenario_path, solution_path, record_path)
    
    if collision_times == None:
      skip_scene.append(scenario_index)
      continue

    # if scenario_index == 1958:
    #   print(">>> ", metrics['fail_rate'], metrics['jerk_loss'], collision_times)
    # else:
    #   print(" ", metrics['fail_rate'], metrics['jerk_loss'], collision_times)

    total_collision_times += collision_times
    # print("metrics['fail_rate']=", metrics['fail_rate'])
    get_list.append([scenario_index, metrics['fail_rate'], metrics['jerk_loss'], collision_times])
    
  print(" ")
  get_array = np.array(get_list)
  norm_array = [1.0]
  for i in range(1, 4):
    _norm = np.max(get_array[:, i])
    if i == 3:
      _norm = max(_norm, 1.0)
    norm_array.append(_norm)
  norm_array = np.array(norm_array)

  index_and_scores = get_array / norm_array
  get_list = [[int(dt[0]), 
               dt[1]*0.6 + dt[2]*0.1 + dt[3]*0.2 # metric coefficents here
    ] for dt in index_and_scores]

  scenario_and_score = sorted(get_list, key=lambda p: p[1], reverse=True) # score from high to low
  num = len(scenario_and_score)

  cond_i = max(0, int(math.ceil(num * 0.3)))
  cond_score = scenario_and_score[cond_i][1]
  max_score = scenario_and_score[0][1]
  scenario_and_score = [dt for dt in scenario_and_score if dt[1] >= cond_score]

  print("Selection condition, score >= {} / max= {}.".format(cond_score, max_score))
  print(" num={}/indexs=".format(len(scenario_and_score)), [dt[0] for dt in scenario_and_score])
  print(" scores=", [round(dt[1], 1) for dt in scenario_and_score])
  print(" skip scenarios=", skip_scene)
