"""
SUMO simulation specific helper methods
"""

__author__ = "Peter Kocsis, Edmond Irani Liu"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Edmond Irani Liu"
__email__ = "edmond.irani@tum.de"
__status__ = "Integration"

import copy
import os
import pickle
from enum import unique, Enum
from math import sin, cos
from typing import Tuple, Dict, Optional

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution import Solution
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from sumocr.interface.ego_vehicle import EgoVehicle
from sumocr.interface.sumo_simulation import SumoSimulation
from sumocr.maps.sumo_scenario import ScenarioWrapper
from sumocr.sumo_config.default import DefaultConfig
from sumocr.sumo_docker.interface.docker_interface import SumoInterface
from sumocr.visualization.video import create_video

from simulation.simulations import load_sumo_configuration, check_trajectories, create_video_for_simulation

from commonroad.prediction.prediction import Occupancy, SetBasedPrediction, TrajectoryPrediction

@unique
class SimulationOption(Enum):
    WITHOUT_EGO = "_without_ego"
    MOTION_PLANNER = "_planner"
    SOLUTION = "_solution"


def simulate_scenario_modified(mode: SimulationOption,
                               conf: DefaultConfig,
                               scenario_wrapper: ScenarioWrapper,
                               scenario_path: str,
                               motion_planner = None,
                               num_of_steps: int = None,
                               planning_problem_set: PlanningProblemSet = None,
                               solution: Solution = None,
                               use_sumo_manager: bool = False) -> Tuple[Scenario, Dict[int, EgoVehicle]]:
    """
    Simulates an interactive scenario with specified mode

    :param mode: 0 = without ego, 1 = with plugged in planner, 2 = with solution trajectory
    :param conf: config of the simulation
    :param scenario_wrapper: scenario wrapper used by the Simulator
    :param scenario_path: path to the interactive scenario folder
    :param motion_planner: your motion planner, which should have .plan() function
    :param num_of_steps: number of steps to simulate
    :param planning_problem_set: planning problem set of the scenario
    :param solution: solution to the planning problem
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :return: simulated scenario and dictionary with items {planning_problem_id: EgoVehicle}
    """

    if num_of_steps is None:
        num_of_steps = conf.simulation_steps

    sumo_interface = None
    if use_sumo_manager:
        sumo_interface = SumoInterface(use_docker=True)
        sumo_sim = sumo_interface.start_simulator()

        sumo_sim.send_sumo_scenario(conf.scenario_name,
                                    scenario_path)
    else:
        sumo_sim = SumoSimulation()

    # initialize simulation
    sumo_sim.initialize(conf, scenario_wrapper, None)

    if mode is SimulationOption.MOTION_PLANNER:
        # simulation with plugged in planner

        def run_simulation():
            ego_vehicles = sumo_sim.ego_vehicles
            for step in range(num_of_steps):
                if use_sumo_manager:
                    ego_vehicles = sumo_sim.ego_vehicles

                # retrieve the CommonRoad scenario at the current time step, e.g. as an input for a prediction module
                current_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
                # >> just record all states in the past: because sumo only get the next_state info...
                # scenario = sumo_sim.commonroad_scenarios_all_time_steps()
                for idx, ego_vehicle in enumerate(ego_vehicles.values()): # len == 1
                    # retrieve the current state of the ego vehicle
                    state_current_ego = ego_vehicle.current_state

                    # ====== plug in your motion planner here
                    # example motion planner which decelerates to full stop
                    if motion_planner != None:
                        print(">> time step = {}".format(sumo_sim.current_time_step))
                        # for dyn_obst in scenario.dynamic_obstacles:
                        #     if isinstance(dyn_obst.prediction,TrajectoryPrediction):
                        #         print("[{}]: {};".format(dyn_obst.obstacle_id, dyn_obst.prediction.trajectory))
                        #         break
                        #     elif isinstance(dyn_obst.prediction,SetBasedPrediction):
                        #         pass
                        #     else:
                        #         if len(current_scenario.dynamic_obstacles) > 0:
                        #             print(current_scenario.dynamic_obstacles)
                        #             print(current_scenario.dynamic_obstacles[0].prediction)
                        #         raise Exception('Unknown dynamic obstacle prediction type: ' + str(type(dyn_obst.prediction))) 

                    next_state = copy.deepcopy(state_current_ego)
                    next_state.steering_angle = 0.0
                    a = -4.0
                    dt = 0.1
                    if next_state.velocity > 0:
                        v = next_state.velocity
                        x, y = next_state.position
                        o = next_state.orientation

                        next_state.position = np.array([x + v * cos(o) * dt, y + v * sin(o) * dt])
                        next_state.velocity += a * dt
                    # ====== end of motion planner

                    # update the ego vehicle with new trajectory with only 1 state for the current step
                    next_state.time_step = 1
                    trajectory_ego = [next_state]
                    ego_vehicle.set_planned_trajectory(trajectory_ego)

                if use_sumo_manager:
                    # set the modified ego vehicles to synchronize in case of using sumo_docker
                    sumo_sim.ego_vehicles = ego_vehicles

                sumo_sim.simulate_step()

        run_simulation()

    # retrieve the simulated scenario in CR format
    simulated_scenario = sumo_sim.commonroad_scenarios_all_time_steps()

    # stop the simulation
    sumo_sim.stop()
    ego_vehicles = {list(planning_problem_set.planning_problem_dict.keys())[0]:
                        ego_v for _, ego_v in sumo_sim.ego_vehicles.items()}

    if use_sumo_manager:
        sumo_interface.stop_simulator()

    return simulated_scenario, ego_vehicles

def simulate_with_planner(interactive_scenario_path: str,
                          motion_planner = None,
                          output_folder_path: str = None,
                          create_video: bool = False,
                          use_sumo_manager: bool = False,
                          create_ego_obstacle: bool = False) \
        -> Tuple[Scenario, PlanningProblemSet, Dict[int, EgoVehicle]]:
    """
    Simulates an interactive scenario with a plugged in motion planner

    :param interactive_scenario_path: path to the interactive scenario folder
    :param output_folder_path: path to the output folder
    :param create_video: indicates whether to create a mp4 of the simulated scenario
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :param create_ego_obstacle: indicates whether to create obstacles from the planned trajectories as the ego vehicles
    :return: Tuple of the simulated scenario, planning problem set, and list of ego vehicles
    """
    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario

    scenario_with_planner, ego_vehicles = simulate_scenario_modified(SimulationOption.MOTION_PLANNER, conf,
                                                                     scenario_wrapper,
                                                                     interactive_scenario_path,
                                                                     motion_planner=motion_planner,
                                                                     num_of_steps=conf.simulation_steps,
                                                                     planning_problem_set=planning_problem_set,
                                                                     use_sumo_manager=use_sumo_manager)
    scenario_with_planner.scenario_id = scenario.scenario_id

    if create_video:
        create_video_for_simulation(scenario_with_planner, output_folder_path, planning_problem_set,
                                    ego_vehicles, SimulationOption.MOTION_PLANNER.value)

    if create_ego_obstacle:
        for pp_id, planning_problem in planning_problem_set.planning_problem_dict.items():
            obstacle_ego = ego_vehicles[pp_id].get_dynamic_obstacle()
            scenario_with_planner.add_objects(obstacle_ego)

    return scenario_with_planner, planning_problem_set, ego_vehicles
