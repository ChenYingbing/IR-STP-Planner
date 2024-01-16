import os
import sys
import argparse
from typing import List, Dict
import random
from utils.colored_print import ColorPrinter
import torch

import thirdparty.config
from envs.commonroad.simulation.utility import save_solution
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.solution import VehicleType, VehicleModel, CostFunction
from commonroad.scenario.scenario import Tag

import envs.config
from envs.config import get_root2folder
from utils.random_seed import set_random_seeds
from utils.file_io import extract_folder_file_list, write_dict2bin
from envs.commonroad.simulation.config import SCENE_DIRS, INVALID_CITY_LIST, INVALID_CITY_KEY, CITY_SCENE_MAX_NUM
from envs.commonroad.simulation.simulation import simulate_with_planner
from planners.speed_profile_generator.generator import SpeedProfileGenerator
from models.prediction.model_factory import PredictionModelFactory
import utils.signal_protection

author = envs.config.AUTHOR_NAME
affiliation = envs.config.AUTHOR_AFFILIATION
source = ''
tags = None

#############################################
# load simulation configurations
import yaml
import conf.__init__
cfg = None
with open(os.path.join(os.path.dirname(conf.__init__.__file__), 
          'planning_commonroad.yaml')) as config_file:
  cfg = yaml.safe_load(config_file)['plan_config']

import matplotlib as mpl
total_simu_steps: int= cfg['simu_steps']
enable_debug_rviz :bool= cfg['enable_debug_rviz']
enable_debug_rviz_step :int= 1000
if enable_debug_rviz:
  enable_debug_rviz_step = cfg['debug_rviz_step_from']
else:
  mpl.use('Agg') # disable matplotlib plot windows

plan_horizon_T :float= cfg['plan_horizon_T']

prediction_mode :str= cfg['prediction_mode']
predict_horizon_L :int= cfg['predict_horizon_L']
predict_traj_mode_num :int= cfg['predict_traj_mode_num']

solution_tag :str= cfg['solu_tag_str']
limit_number_of_scenarios :int = cfg['limit_number_of_scenarios']
involved_subsequent_scenarios :bool = cfg['involved_subsequent_scenarios']
involved_scenarios :list= cfg['involved_scenarios']

#############################################
if __name__ == '__main__':  
  parser = argparse.ArgumentParser()
  parser.add_argument("--seeds", type=int, help="random seeds", default=0)
  parser.add_argument("--model_path", type=str, help="path to prediction network model", required=True)
  parser.add_argument("--check_point", type=str, help="check point index (string)", default='best')
  parser.add_argument("--create_video", type=int, help="enable create video or not", default=1)

  # optional arguments to replace those in yaml files
  parser.add_argument("--planner_type", type=str, default=None)
  parser.add_argument("--solu_tag_str", type=str, default=None)
  parser.add_argument("--prediction_mode", type=str, default=None)
  parser.add_argument("--predict_traj_mode_num", type=int, default=None)
  parser.add_argument("--algo_variable1", type=float, default=None)
  parser.add_argument("--algo_variable2", type=float, default=None)
  parser.add_argument("--algo_variable3", type=float, default=None)
  parser.add_argument("--ignore_influ_cons", type=int, default=None)
  parser.add_argument("--st_coefficents", type=int, default=None)
  parser.add_argument("--acc_mode", type=str, default=None)
  parser.add_argument("--acc_const_value", type=float, default=None)
  parser.add_argument("--reaction_traj_mode", type=str, default=None)
  parser.add_argument("--is_exp_mode", help="is experiment mode", type=int, default=None)

  args = parser.parse_args()

  #############################################
  # replace the original config in yaml file
  if args.planner_type:
    cfg['planner_type'] = args.planner_type
  if args.solu_tag_str:
    solution_tag = args.solu_tag_str
  if args.prediction_mode:
    prediction_mode = args.prediction_mode
  if args.predict_traj_mode_num:
    predict_traj_mode_num = args.predict_traj_mode_num
  if args.algo_variable1:
    cfg['reaction_config']['algo_variable1'] = args.algo_variable1
  if args.algo_variable2:
    cfg['reaction_config']['algo_variable2'] = args.algo_variable2
  if args.algo_variable3:
    cfg['reaction_config']['algo_variable3'] = args.algo_variable3
  if args.ignore_influ_cons:
    cfg['reaction_config']['reaction_conditions']['ignore_influ_cons'] = (args.ignore_influ_cons > 0)
  if args.st_coefficents:
    cfg['reaction_config']['reaction_conditions']['st_coefficents'] = args.st_coefficents
  if args.acc_mode:
    cfg['reaction_config']['reaction_conditions']['acc_mode'] = args.acc_mode
  if args.acc_const_value:
    cfg['reaction_config']['reaction_conditions']['acc_const_value'] = args.acc_const_value
  if args.reaction_traj_mode:
    cfg['reaction_config']['reaction_conditions']['traj_mode'] = args.reaction_traj_mode
  
  pickout_intersections = [
    643, 644, 645, 646, 647, 648, 649, 652, 653, 654, 656, 659, 660, 661, 704, 705,  
    709, 710, 711, 712, 715, 718, 719, 720, 722, 729, 732, 736, 740, 743, 747, 750,
    751, 757, 759, 760, 761, 762, 763, 764, 765, 767, 770, 798, 799, 800, 803, 804, 
    806, 808, 810, 811, 813, 816, 828, 835, 836, 838, 840, 841, 843, 844, 847, 874, 
    881, 899, 906, 907, 917, 930, 933, 1062, 1168, 1173, 1175, 1242
  ]

  if args.is_exp_mode:
    if args.is_exp_mode > 0:
      enable_debug_rviz = False
      enable_debug_rviz_step = 1000

    involved_scenarios = []
    if args.is_exp_mode == 1:
      # all scenarios
      involved_scenarios = [] # all are envolved
    elif args.is_exp_mode == 2:
      # scenarios at intersections
      involved_scenarios = pickout_intersections
    elif args.is_exp_mode == 3:
      # a test scenario
      involved_scenarios = [643]
    else:
      raise NotImplementedError("exp mode = %d, not implemented")
  print("EXP MODE == {}, with {} scenarios.".format(args.is_exp_mode, len(involved_scenarios)))
  print("SCENARIOS=", involved_scenarios)

  #############################################
  path_solutions = get_root2folder(
    envs.config.get_dataset_exp_folder('commonroad', "exp_plan/{}".format(solution_tag)), "solutions")
  path_video = get_root2folder(
    envs.config.get_dataset_exp_folder('commonroad', "exp_plan/{}".format(solution_tag)), "videos")
  path_scenarios_simulated = get_root2folder(
    envs.config.get_dataset_exp_folder('commonroad', "exp_plan/{}".format(solution_tag)), "result_scenarios")

  scene_paths: List[str] = []              # list of scene path
  city_indexs: Dict[str, List[int]] = {}   # map from city > list of scene_path_indexs
  for i, scene_dir in enumerate(SCENE_DIRS):
    scenario_list = extract_folder_file_list(scene_dir)
    for scenario in scenario_list:
      scene_paths.append(os.path.join(scene_dir, scenario))
      this_index = len(scene_paths) - 1

      tag = scenario.split('-')
      city = tag[0]
      if (not city in INVALID_CITY_LIST) and (not INVALID_CITY_KEY in city):
        if not city in city_indexs:
          city_indexs[city] = []
        city_indexs[city].append(this_index)

  # print("city_indexs=")
  # print(city_indexs)
  # print(" ")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  pmodel = PredictionModelFactory.produce(
    device, args.model_path, args.check_point,
    prediction_mode=prediction_mode,
    predict_horizon_L=predict_horizon_L,
    predict_traj_mode_num=predict_traj_mode_num)

  #############################################
  # Traverse the city and scenes
  count :int= 0
  solu_index :int= 0
  involved_scenarios_has_occured :bool= False
  for city, indexs in city_indexs.items():
    print(" \nCity:{} with indexs num={}.".format(city, len(indexs)))
    # print("Indexs", indexs)
    assert not city in INVALID_CITY_LIST, "Fatal Error."
    # random.shuffle(indexs) # shuffle the list
    one_city_scene_num = 0

    for index in indexs:
      if (len(involved_scenarios) > 0) and (not index in involved_scenarios):
        # when len(involved_scenarios) > 0, only run involved_scenarios list scenarios
        if involved_subsequent_scenarios:
          if not involved_scenarios_has_occured:
            continue
          else:
            pass # not skip this scenario
        else:
          continue
      else:
        involved_scenarios_has_occured = True

      # other conditions
      if count >= limit_number_of_scenarios:
        break

      if (one_city_scene_num > CITY_SCENE_MAX_NUM) and (not index in pickout_intersections):
        # skip when one city contains too much scenarios
        continue
      
      path_scenario = scene_paths[index]
      scenario_name = path_scenario.split('/')[-1]
      # # debug specific scenario
      # if not 'DEU_Muc-2_1_I-1-1' in scenario_name:
      #   continue

      is_legal = False
      file_list = extract_folder_file_list(path_scenario)
      for fname in file_list:
        # check if exists compulsory config file
        if fname == 'simulation_config.p':
          is_legal = True
          break

      if is_legal:
        set_random_seeds(args.seeds)

        # new a planner
        planner = SpeedProfileGenerator(
          plan_horizon_T=plan_horizon_T,
          predict_traj_mode_num=cfg['predict_traj_mode_num'],
          planner_type=cfg['planner_type'],
          reaction_config=cfg['reaction_config'])
  
        success_flag, probs_set, ego_vehicles, scenario_with_planner, dict_records =\
          simulate_with_planner(scenario_index=index, 
                                interactive_scenario_path=path_scenario,
                                motion_planner=planner,
                                prediction_model=pmodel,
                                max_num_of_steps=total_simu_steps,
                                create_video=(args.create_video > 0),
                                video_folder_path=path_video,
                                enable_debug_rviz_step=enable_debug_rviz_step,
                                random_seed=args.seeds)

        if not success_flag:
          continue # run the later codes only when success simulated

        ColorPrinter.print('yellow', 'Save to the following folders')
        print(">> Solutions:", path_solutions)
        print(">> Videos:", path_video)
        print(">> Result_scenes", path_scenarios_simulated)
        print(" ")

        # Write the simulation results to CommonRoad xml file
        fw = CommonRoadFileWriter(scenario_with_planner, probs_set, author, affiliation, source, tags)
        fw.write_to_file(os.path.join(path_scenarios_simulated, 
                         "[{}]_{}_results.xml".format(solu_index, scenario_name)), 
                         OverwriteExistingFile.ALWAYS)

        # save the planned trajectory to solution file
        # @note vehicle_type, vehicle_model, cost_function are set arbitrarily,
        #   since their values will not used in solution evaluation.
        vehicle_type = VehicleType.FORD_ESCORT
        vehicle_model = VehicleModel.KS
        cost_function = CostFunction.TR1

        save_solution(scenario_with_planner, probs_set, ego_vehicles, vehicle_type, 
                      vehicle_model, cost_function, path_solutions, 
                      filename='[{}]solu_{}.xml'.format(solu_index, scenario_name), overwrite=True)
        write_dict2bin(
          dict_records,
          os.path.join(path_solutions, 'plan_records[{}].bin'.format(solu_index)),
          verbose=False
        )
        solu_index += 1
        # print("Process index", index, one_city_scene_num, count)

      # add operations
      one_city_scene_num+=1
      count += 1

      # debug info
      print("Process index", index, one_city_scene_num, count)
      print("Simulating scene={}/{}.".format(one_city_scene_num, len(indexs)))

  print("")
  print("All simulation finished")
