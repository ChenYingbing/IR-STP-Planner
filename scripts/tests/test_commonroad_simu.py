import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import envs.config
import thirdparty.config
from simulation.simulations import simulate_without_ego, simulate_with_solution
from simulation.utility import visualize_scenario_with_trajectory, save_solution
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.solution import CommonRoadSolutionReader, VehicleType, VehicleModel, CostFunction
from commonroad.scenario.scenario import Tag

from envs.commonroad.simulation.simulation_demo import simulate_with_planner

'''
This is a test of commonroad interactive simulation with sumo
'''

# example scenario folder: envs.config.COMMONROAD_EXAMPLE_SCENARIOS_PATH
# config the path envs
name_scenario = "DEU_Cologne-63_5_I-1"
path_scenario = os.path.join(envs.config.COMMONROAD_EXAMPLE_SCENARIOS_PATH, name_scenario)

path_solutions = envs.config.get_dataset_exp_folder('commonroad', "outputs/solutions")
path_video = envs.config.get_dataset_exp_folder('commonroad', "outputs/videos")
path_scenarios_simulated = envs.config.get_dataset_exp_folder('commonroad', "outputs/simulated_scenarios")

author = envs.config.AUTHOR_NAME
affiliation = envs.config.AUTHOR_AFFILIATION
source = ''
tags = {Tag.HIGHWAY, Tag.INTERSTATE}

vehicle_type = VehicleType.FORD_ESCORT
vehicle_model = VehicleModel.KS
cost_function = CostFunction.TR1

mode = 1
if mode == 1:
  # run simulation, a video animation of the simulation is stored in the end
  scenario_without_ego, pps = simulate_without_ego(interactive_scenario_path=path_scenario,
                                                   output_folder_path=path_video,
                                                   create_video=False)
  # write simulated scenario to CommonRoad xml file
  fw = CommonRoadFileWriter(scenario_without_ego, pps, author, affiliation, source, tags)
  fw.write_to_file(os.path.join(path_scenarios_simulated, name_scenario + "_no_ego.xml"), 
                   OverwriteExistingFile.ALWAYS)
elif mode == 2:
  class your_motion_planner:
    def __init__(self):
      pass
  planner = your_motion_planner()

  # run simulation, an animation of the simulation is stored in the end
  scenario_with_planner, pps, ego_vehicles_planner =\
    simulate_with_planner(interactive_scenario_path=path_scenario,
                          motion_planner=planner,
                          output_folder_path=path_video,
                          create_video=False)

  # write simulated scenario to CommonRoad xml file
  if scenario_with_planner:
    # write simulated scenario to file
    fw = CommonRoadFileWriter(scenario_with_planner, pps, author, affiliation, source, tags)
    fw.write_to_file(os.path.join(path_scenarios_simulated, name_scenario + "_planner.xml"), 
                     OverwriteExistingFile.ALWAYS)
    
    # save the planned trajectory to solution file
    save_solution(scenario_with_planner, pps, ego_vehicles_planner, vehicle_type, 
                  vehicle_model, cost_function,
                  path_solutions, overwrite=True)
elif mode == 3:
  name_solution = "DEU_Cologne-63_5_I-1"
  solution = CommonRoadSolutionReader.open(os.path.join(path_solutions, name_solution + ".xml"))

  # run simulation, a video of the simulation is stored in the end
  scenario_with_solution, pps, ego_vehicles_solution = simulate_with_solution(interactive_scenario_path=path_scenario,
                                                                              output_folder_path=path_video,
                                                                              solution=solution,
                                                                              create_video=False)
  # write simulated scenario to CommonRoad xml file
  if scenario_with_solution:
      # write simulated scenario to file
      fw = CommonRoadFileWriter(scenario_with_solution, pps, author, affiliation, source, tags)
      fw.write_to_file(os.path.join(path_scenarios_simulated, name_scenario + "_solution.xml"), 
                                    OverwriteExistingFile.ALWAYS)
