import os
import envs.config
from typing import Tuple
import sys
import envs.config
from typing import List, Dict
import traceback

import thirdparty.config
from simulation.utility import save_solution
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.solution import CommonRoadSolutionReader, VehicleType, VehicleModel, CostFunction
from commonroad.scenario.scenario import Tag
from simulation.simulations import load_sumo_configuration, simulate_without_ego
from utils.file_io import extract_folder_file_list

from envs.commonroad.simulation.config import SCENE_DIRS, INVALID_CITY_LIST, INVALID_CITY_KEY, CITY_SCENE_MAX_NUM

### ### ### ### ### ### ### ### 
from commonroad.common.file_reader import CommonRoadFileReader

### ### ### ### ### ### ### ### 
import argparse
from utils.random_seed import set_random_seeds
import random

if __name__ == '__main__':  
  parser = argparse.ArgumentParser()
  parser.add_argument("--seeds", type=int, help="random seeds", default=0)
  parser.add_argument("--mode", type=str, help="mode pattern", default="extract")
  args = parser.parse_args()

  path_solutions = envs.config.get_dataset_exp_folder('commonroad', "outputs/solutions")
  path_video = envs.config.get_dataset_exp_folder('commonroad', "outputs/videos")
  path_scenarios_simulated = envs.config.get_dataset_exp_folder('commonroad', "simulated_scenarios")

  set_random_seeds(args.seeds)

  if args.mode == 'extract':
    # each city maximum sample how much scenes.
    source = ''
    tags = {Tag.URBAN}

    # vehicle_type = VehicleType.FORD_ESCORT
    # vehicle_model = VehicleModel.KS

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
    
    for city, indexs in city_indexs.items():
      # name_scenario = "DEU_Cologne-63_5_I-1"
      # path_scenario = os.path.join(envs.config.COMMONROAD_EXAMPLE_SCENARIOS_PATH, name_scenario)
      #
      print("City:{} with indexs num={}.".format(city, len(indexs)))
      assert not city in INVALID_CITY_LIST, "Fatal Error."
      random.shuffle(indexs) # shuffle the list
      scene_add_num = 0

      for index in indexs:
        path_scenario = scene_paths[index]
        try:
          # @note using try because sumo simulation may not always work successfully.
          scenario_without_ego, pps = simulate_without_ego(interactive_scenario_path=path_scenario,
                                                           output_folder_path=path_video,
                                                           create_video=True)
          # write simulated scenario to CommonRoad xml file
          fw = CommonRoadFileWriter(scenario_without_ego, pps, 
                                    envs.config.AUTHOR_NAME, envs.config.AUTHOR_AFFILIATION, source, tags)
          fw.write_to_file(os.path.join(path_scenarios_simulated, 
                           city + "-T-{}.xml".format(scene_add_num)), 
                           OverwriteExistingFile.ALWAYS)

          scene_add_num+=1
        except Exception as einfo:
          print(einfo)

        print("Extract scene amount={}/{}.".format(scene_add_num, CITY_SCENE_MAX_NUM))
        if scene_add_num >= CITY_SCENE_MAX_NUM:
          break

  else:
    print("Warning, the value of args.mode should be either 'extract' or 'check'.")

