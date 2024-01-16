# modified from commonroad-interactive-scenarios.simulation.simulations.py

import copy
import imp
import os
import pickle
from enum import unique, Enum
from math import sin, cos
from typing import Tuple, Dict, Optional, List, Any
import numpy as np
import logging
import matplotlib.pyplot as plt
import math

from envs.commonroad.simulation.sumo_simulation import SumoSimulation
from envs.commonroad.simulation.utility import agent_limited_length, agent_limited_width

# third party
import thirdparty.config
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution import Solution
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.common.util import Interval
from commonroad.scenario.obstacle import ObstacleRole
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.planning.goal import GoalRegion
from sumocr.interface.ego_vehicle import EgoVehicle
from sumocr.maps.sumo_scenario import ScenarioWrapper
from sumocr.sumo_config.default import DefaultConfig
from sumocr.sumo_docker.interface.docker_interface import SumoInterface
from sumocr.visualization.video import create_video
from simulation.simulations import load_sumo_configuration, check_trajectories, create_video_for_simulation

# planner & predicion module
from planners.planner import BasicPlanner
from models.prediction.model import BasicPredictionModel
from preprocessor.commonroad.vectorizer import CommonroadVectorizer
from preprocessor.commonroad.graph import CommonroadGraphProcessor
import commonroad
from commonroad.scenario.obstacle import ObstacleType
from utils.geometry import get_box

# utils
from envs.config import get_root2folder
import envs.__init__
from envs.navigation_map import NavigationMap
from utils.file_io import read_yaml_config
from utils.transform import XYYawTransform
from envs.commonroad.simulation.rviz import SimulationVisualizer
from type_utils.agent import EgoAgent
import envs.commonroad.__init__

from utils.colored_print import ColorPrinter

FLAG_SKIP_SCENARIO :int = 0

def simulate_with_prediction(conf: DefaultConfig,
                             random_seed: int,
                             preprocess_config: Dict,
                             motion_planner: BasicPlanner,
                             prediction_model: BasicPredictionModel,
                             scenario_wrapper: ScenarioWrapper,
                             scenario_index: int,
                             scenario_path: str,
                             num_of_steps: int = None,
                             planning_problem_set: PlanningProblemSet = None,
                             use_sumo_manager: bool = False,
                             create_video: bool = False,
                             video_path: str = None,
                             enable_debug_rviz_step: int = 1e+3) -> Tuple[bool, Scenario, Dict[int, EgoVehicle], Dict]:
    """
    Simulates an interactive scenario with specified motion planner

    :param conf: config of the simulation
    :param random_seed: random seed setting
    :param motion_planner: planner class with port function plan()
    :param prediction_model: trained network-based prediction model
    :param scenario_wrapper: scenario wrapper used by the Simulator
    :param scenario_index: index of the scenario folder
    :param scenario_path: path to the interactive scenario folder
    :param num_of_steps: number of steps to simulate
    :param planning_problem_set: planning problem set of the scenario
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :param enable_debug_rviz_step: after which step, the debug rviz is enabled
    :return: dictionary items 
             {success, scenario, EgoVehicle, Dict_of_records}
    """

    ### ### ### ### ### ### ### 
    if num_of_steps is None:
        num_of_steps = conf.simulation_steps

    sumo_interface = None
    if use_sumo_manager:
        sumo_interface = SumoInterface(use_docker=True)
        sumo_sim = sumo_interface.start_simulator()

        sumo_sim.send_sumo_scenario(conf.scenario_name,
                                    scenario_path)
    else:
        sumo_sim = SumoSimulation(set_random_seed=random_seed)

    # @example:
    #   scneario_path = 'commonroad-scenarios/scenarios/interactive/hand-crafted/DEU_A9-1_1_I-1-1/DEU_A9-1_1_I-1-1'
    #   scenario_folder = 'hand-crafted/DEU_A9-1_1_I-1-1/DEU_A9-1_1_I-1-1'
    #   path2scenario_folder = ${video_path}/scenario_folder/
    remove_len = len(os.path.dirname(os.path.dirname(os.path.dirname(scenario_path))))
    scenario_str = scenario_path[remove_len+1:] # +1 is to remove '\'

    # initialize simulation
    # @note in commonroad interactive simulation tutorial, planning_problem_set = None, 
    #       which will cause non-ego vehicle in the scenario.
    #       It can be solved by using planning_problem_set=planning_problem_set
    # print("check problem set keys", planning_problem_set.planning_problem_dict.keys())
    # try:
    # fix a bug of key error in lib sumo_simulation
    print("Simulation start: {}".format(conf.scenario_name))
    # [modified]
    try:
      for ckey in conf.veh_params.keys(): # ['width', 'length', 'minGap']:
          conf.veh_params[ckey][ObstacleType.MOTORCYCLE] = conf.veh_params[ckey]['motorcycle']
      for ckey in conf.veh_params['length']:
          conf.veh_params['length'][ckey] = min(conf.veh_params['length'][ckey], agent_limited_length())
      for ckey in conf.veh_params['width']:
          conf.veh_params['width'][ckey] = min(conf.veh_params['width'][ckey], agent_limited_width())
    except:
      pass # key error skips.

    # # [modified]
    # # default= {'motorcycle': 6, <ObstacleType.CAR: 'car'>: 7.5, <ObstacleType.TRUCK: 'truck'>: 4, <ObstacleType.BUS: 'bus'>: 4, <ObstacleType.BICYCLE: 'bicycle'>: 3, <ObstacleType.PEDESTRIAN: 'pedestrian'>: 2, <ObstacleType.MOTORCYCLE: 'motorcycle'>: 6}
    # for ckey in conf.veh_params['decel']:
    #     conf.veh_params['decel'][ckey] = 6.0 # all type of agent unified dcc  
    # # default= {'motorcycle': 2.5, <ObstacleType.CAR: 'car'>: 2.9, <ObstacleType.TRUCK: 'truck'>: 1.3, <ObstacleType.BUS: 'bus'>: 1.2, <ObstacleType.BICYCLE: 'bicycle'>: 1.2, <ObstacleType.PEDESTRIAN: 'pedestrian'>: 1.5, <ObstacleType.MOTORCYCLE: 'motorcycle'>: 2.5}
    # for ckey in conf.veh_params['accel']:
    #     conf.veh_params['accel'][ckey] = 2.0 # all type of agent unified acc

    # # default= {'motorcycle': 2.5, <ObstacleType.CAR: 'car'>: 1.0, <ObstacleType.TRUCK: 'truck'>: 2.5, <ObstacleType.BUS: 'bus'>: 2.5, <ObstacleType.BICYCLE: 'bicycle'>: 1.0, <ObstacleType.PEDESTRIAN: 'pedestrian'>: 0.25, <ObstacleType.MOTORCYCLE: 'motorcycle'>: 2.5}
    # for ckey in conf.veh_params['minGap']:
    #     conf.veh_params['minGap'][ckey] = 2.0 # all type of agent has same min gap

    # print("conf.veh_params")
    # print(conf.veh_params['length'])
    # print(conf.veh_params['width'])
    # # print(conf.veh_params['minGap'])
    # # print(conf.veh_params['accel'])
    # # print(conf.veh_params['decel'])
    # # print(conf.veh_params['maxSpeed'])
    # print(" ")

    # ######################################################################
    # # debug specific scenario
    # global FLAG_SKIP_SCENARIO
    # if (conf.scenario_name != 'DEU_Frankfurt-100_12_I-1'):
    #   print("skip", conf.scenario_name)
    # else:
    #   FLAG_SKIP_SCENARIO = 1

    # if FLAG_SKIP_SCENARIO > 0:
    #   FLAG_SKIP_SCENARIO += 1
    #   FLAG_SKIP_SCENARIO = min(FLAG_SKIP_SCENARIO, 100)
    # if FLAG_SKIP_SCENARIO <= 0:
    #   return False, None, None, [0, 0]

    ######################################################################
    config_file_path =\
        os.path.join(os.path.dirname(envs.commonroad.__init__.__file__), 
                        'conf/simuconfig.sumocfg')
    try:
        sumo_sim.initialize(conf, 
            scenario_wrapper, planning_problem_set, 
            additional_file_path=config_file_path)
    except:
        # @problem libsumo.libsumo.TraCIException: 
        #   Departure speed for vehicle 'egoVehicle0' is too high for the vehicle type 'passenger_static'.
        ColorPrinter.print('yellow', "Initialization fail, skip scenario {}.".format(scenario_str))
        return False, None, None, [0, 0]    

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("conf", conf.veh_params.keys())
    print(conf.veh_params['width'].keys())

    plan_problem: PlanningProblem = None
    for _, problem in planning_problem_set.planning_problem_dict.items():
        plan_problem = problem # return first problem
        break

    ### ### ### ### ### ### ###
    visual = SimulationVisualizer()
    path2scenario_folder = ""
    if create_video:
        # create scenario folder
        scenario_folder = str(scenario_index) + '_' + scenario_str.split('/')[-1]
        path2scenario_folder = get_root2folder(video_path, scenario_folder)
        # write information to txt
        txt_path = os.path.join(path2scenario_folder, '0_scenario_info.txt')
        txt_file = open(txt_path, 'w')
        txt_file.write(scenario_str)
        txt_file.close()

    ### ### ### ### ### ### ### 
    if len(plan_problem.goal.state_list) == 0:
        ColorPrinter.print('yellow', "Goal num == 0, skip scenario {}.".format(scenario_str))
        return False, None, None, [0, 0]

    # prepare global route, ...
    # print("Planning problem[{}]: goal={}, {}.".format(
    #     plan_problem.planning_problem_id,
    #     len(plan_problem.goal.state_list), 
    #     plan_problem.goal.lanelets_of_goal_position)
    # )
    # : plan_problem.initial_state type is: State
    # : plan_problem.goal type is: GoalRegion
    start = plan_problem.initial_state
    goal = plan_problem.goal.state_list[0]

    init_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
    nav_map = NavigationMap()
    nav_map.InitWithLaneletNet(init_scenario.lanelet_network, verbose=False)
    nav_start = XYYawTransform(x=start.position[0], 
                               y=start.position[1],
                               yaw_radian=start.orientation)
    # init_scenario.lanelet_network.cleanup_traffic_light_references()

    # if not isinstance(goal, State):
    goal_pos = None
    goal_orientation = None
    if not hasattr(goal, 'position'):
        # only has attribute .time_step
        # Require the av surrive until certain time. <commonroad.common.util.Interval >
        ColorPrinter.print("blue", "IsInstance of Interval")
        nav_goal = nav_map.DepthFirstSearchGoal(nav_start)
    else:
        if isinstance(goal.position, commonroad.geometry.shape.Rectangle):
            ColorPrinter.print("blue", "IsInstance of Rectangle")
            goal_pos = goal.position
            goal_orientation = goal.position.orientation
        elif isinstance(goal.position, commonroad.geometry.shape.ShapeGroup):
            ColorPrinter.print("blue", "IsInstance of ShapeGroup")
            # goal.position.shapes[0]: commonroad.geometry.shape.Polygon
            goal_pos = goal.position.shapes[0]
            goal_orientation = nav_map.GetValidGoalOrientation(
                nav_start, goal_pos.center[0], goal_pos.center[1])
        else:
            ColorPrinter.print("blue", "IsInstance Unkown")
            ColorPrinter.print('yellow', 
                "goal.position type={}, with {}".format(type(goal.position), goal.position))

        nav_goal = XYYawTransform(x=goal_pos.center[0], 
                                  y=goal_pos.center[1],
                                  yaw_radian=goal_orientation)

    has_route = nav_map.InitRoute(nav_start, nav_goal, verbose=True)
    if has_route == False:
        ColorPrinter.print('yellow', "Invalid route, skip scenario {}.".format(scenario_str))
        return False, None, None, [0, 0]
    # nav_map.plot_lane_graph(save_path=None, fig_dpi=300)

    # simulation with plugged in planner
    ego_vehicles = sumo_sim.ego_vehicles
    ego_agent = None

    # fextractor = FeatureExtractor()
    preprocess_mode = preprocess_config['process_mode']
    fextractor = None
    is_graph_mode= False
    if preprocess_mode == 'graph':
        fextractor = CommonroadGraphProcessor(args=preprocess_config)
        is_graph_mode = True
    elif preprocess_mode == 'vectorize':
        fextractor = CommonroadVectorizer(args=preprocess_config)
    else:
        raise ValueError(f"Unkown preprocessor mode = {preprocess_mode}")

    config_slowdown_step :int = 4 # before which step, excute pure slowdown operation
    plan_success_steps :int = 0
    total_simu_steps :int = 0
    list_plan_visited_edges_count = []
    list_of_time_costs = []
    list_of_obs_num = []
    for step in range(num_of_steps):
        simulation_not_fully_started = step <= config_slowdown_step

        print("\n" + ">"*40)
        ColorPrinter.print('yellow', "[{}] simulating, step={}/{}, with rviz_cond_step={}.".format(
            scenario_str.split('/')[-1], step, num_of_steps, enable_debug_rviz_step))
        if use_sumo_manager:
            ego_vehicles = sumo_sim.ego_vehicles

        # @note current_scenario only have one time step. However, prediction 
        #       module requires past infos. As a result, here use 
        #       commonroad_scenarios_all_time_steps() instead.
        # current_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
        current_scenario = sumo_sim.commonroad_scenarios_all_time_steps()

        num_egos = len(ego_vehicles.values())
        if (num_egos != 1):
            print("Disable simulation, number of ego vehicle={}, !=1.".format(num_egos))
            # TODO: this is a simple version, which is unable to deal with multi-ego vehicles.
            break

        # extract ego_vehicle
        ego_vehicle: EgoVehicle= None
        for _, vehicle in enumerate(ego_vehicles.values()):
            ego_vehicle = vehicle

        # get prediction: agents_pred_trajs
        # @note here enable_rviz is to visualize the rastered image for
        #       each agent being predicted.
        view_limits = visual.get_xy_limits(ego_vehicle)
        
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
        # agent_list, agent_states, extract_dt =\
        #     fextractor.extract_scenario(
        #         current_scenario, step, view_limits, enable_rviz=False)

        fextractor.init_scenario(current_scenario, enable_rviz=False)
        fextractor.init_frame_from_to(step, step)
        agent_list, agent_states, agent_inputs = [], [], []

        obs_list = current_scenario.obstacles_by_position_intervals(
            [Interval(view_limits[0][0], view_limits[0][1]), 
            Interval(view_limits[1][0], view_limits[1][1])], 
            tuple(ObstacleRole), time_step=step)
        for dyn_obst in obs_list: # scenario.dynamic_obstacles
            agent_idx = dyn_obst.obstacle_id

            get_inputs = fextractor.get_inputs(idx=-1, agent_idx=agent_idx)
            if is_graph_mode:
                # init node is essential for graph representation
                get_inputs['is_reference'] = True
                get_inputs['init_node'] =\
                    fextractor.get_initial_node(get_inputs['map_representation'])

            state = dyn_obst.state_at_time(step)
            xyyaw = XYYawTransform(
                x=state.position[0], 
                y=state.position[1],
                yaw_radian=state.orientation)

            obstacle_shape = dyn_obst.obstacle_shape
            agent_list.append({
                'idx': agent_idx,
                'length': obstacle_shape.length,
                'width': obstacle_shape.width,
                'shape': get_box(
                    centroid=obstacle_shape.center, 
                    yaw=obstacle_shape.orientation,
                    extent=[obstacle_shape.width, 
                        obstacle_shape.length]),
                'xyyaw': xyyaw,
                'velocity': state.velocity,
                'acceleration': state.acceleration,
            })
            agent_states.append(xyyaw)
            agent_inputs.append(get_inputs)
            # print("get_inputs keys=", get_inputs.keys())
            # print("get map_representation keys=", get_inputs['map_representation'].keys())

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
        state_current_ego = ego_vehicle.current_state
        ego_xyyaw = XYYawTransform(
            x=state_current_ego.position[0],
            y=state_current_ego.position[1],
            yaw_radian=state_current_ego.orientation)

        obs_num = 0
        agents_pred_trajs = []
        if len(agent_list) > 0:
            agents_pred_trajs =\
                prediction_model.predict(
                    ego_xyyaw, agent_list, agent_states, agent_inputs)
            
            obs_num = len(agent_states)
            # for agent_xyyaw in agent_states:
            #     dx = agent_xyyaw._x - ego_xyyaw._x
            #     dy = agent_xyyaw._y - ego_xyyaw._y
            #     if math.sqrt(dx*dx + dy*dy) <= 50.0:
            #         obs_num += 1
        if (simulation_not_fully_started == False):
            list_of_obs_num.append(obs_num)

        # retrieve the current state of the ego vehicle
        if ego_agent == None:
            ego_agent = EgoAgent(
                id=ego_vehicle.id,
                width=ego_vehicle.width,
                length=ego_vehicle.length,
                time_step_dt=current_scenario.dt,
            )

        # ====== plug in your motion planner here
        if step == 0:
            # init at first step
            state_current_ego.steering_angle = 0.0
            state_current_ego.acceleration = 0.0

        try: # some scenario do not have yaw_rate..
          ego_agent.set_state(
            time_step_idx=step,
            pos_x=state_current_ego.position[0],
            pos_y=state_current_ego.position[1],
            orientation=state_current_ego.orientation,
            steering_radian=state_current_ego.steering_angle,
            velocity=state_current_ego.velocity,
            yaw_rate=state_current_ego.yaw_rate, 
            acceleration=state_current_ego.acceleration
          )
        except:
          ego_agent.set_state(
            time_step_idx=step,
            pos_x=state_current_ego.position[0],
            pos_y=state_current_ego.position[1],
            orientation=state_current_ego.orientation,
            steering_radian=state_current_ego.steering_angle,
            velocity=state_current_ego.velocity,
            yaw_rate=0.0,
            acceleration=state_current_ego.acceleration
          )

        planning_result, time_dict =\
            motion_planner.plan(
                nav_map=nav_map,
                time_step=step,
                agents=agent_list,
                agent_predictions=agents_pred_trajs,
                ego_agent=ego_agent,
                require_slowdown=simulation_not_fully_started,
                enable_rviz_step=enable_debug_rviz_step
            )
        print("ego_agent states:", ego_agent.states[step])
        print("with width, length=", ego_agent.info['width'], ego_agent.info['length'])
        # print(">>>>>>>>>>>> GGGGGGGGGGGGGGGGGGGGGGGGGG ")
        # print(current_scenario.lanelet_network.traffic_lights)
        # for ss_list in current_scenario.lanelet_network.traffic_signs:
        #     for ss in ss_list.traffic_sign_elements:
        #         print(ss.traffic_sign_element_id)

        next_state = copy.deepcopy(state_current_ego)
        tva_xyyawcur_array = None
        if planning_result['has_result']:
            tva_xyyawcur_array = planning_result['tva_xyyawcur_array']
            plan_success_steps += 1
            
            if (simulation_not_fully_started == False):
                list_plan_visited_edges_count.append(planning_result['edge_counts'])

        else:
            tva_xyyawcur_array = planning_result['tva_xyyawcur_array'] # still has values
            if (simulation_not_fully_started == True) or\
               (ego_agent.states[step]['velocity'] < 1e-2):
                # cond1: (not start) deemed as true
                # cond2: is already stop deemd as true
                plan_success_steps += 1

            if (simulation_not_fully_started == False):
                list_plan_visited_edges_count.append(planning_result['edge_counts'])

        need_direct_brake = True
        if isinstance(tva_xyyawcur_array, np.ndarray):
            if tva_xyyawcur_array.shape[0] >= 2:
                next_tva_xyyawcur = tva_xyyawcur_array[1, :]
                
                next_state.steering_angle = 0.0
                next_state.position[0] = next_tva_xyyawcur[3]
                next_state.position[1] = next_tva_xyyawcur[4]
                next_state.orientation = next_tva_xyyawcur[5]
                next_state.velocity = next_tva_xyyawcur[1]
                next_state.acceleration = next_tva_xyyawcur[2]
                need_direct_brake = False
        
        if need_direct_brake:
            # example motion planner which decelerates to full stop
            next_state.steering_angle = 0.0
            a = -4.0
            dt = 0.1
            if next_state.velocity > 0:
                v = next_state.velocity
                x, y = next_state.position
                o = next_state.orientation

                next_state.position = np.array([x + v * cos(o) * dt, y + v * sin(o) * dt])
                next_state.velocity += a * dt
                next_state.acceleration = a

        if (simulation_not_fully_started == False):
            list_of_time_costs.append(time_dict)
        
        total_simu_steps += 1
        
        # ====== end of motion planner

        # update the ego vehicle with new trajectory with only 1 state for the current step
        next_state.time_step = 1
        trajectory_ego = [next_state]
        ego_vehicle.set_planned_trajectory(trajectory_ego)

        # rviz
        if create_video:
            get_behaviors = motion_planner.get_ref_behaviors()
            visual.save_fig(
                path2scenario_folder,
                current_scenario, step, ego_vehicle, view_limits,
                get_behaviors, planning_result, agent_list, agents_pred_trajs)

        if use_sumo_manager:
            # set the modified ego vehicles to synchronize in case of using sumo_docker
            sumo_sim.ego_vehicles = ego_vehicles

        sumo_sim.simulate_step()

    ### ### ### ### ### ### ### 
    print("Simulation quit.")
    # retrieve the simulated scenario in CR format
    simulated_scenario = sumo_sim.commonroad_scenarios_all_time_steps()

    # stop the simulation
    sumo_sim.stop()
    ego_vehicles = {list(planning_problem_set.planning_problem_dict.keys())[0]:
                        ego_v for _, ego_v in sumo_sim.ego_vehicles.items()}

    if use_sumo_manager:
        sumo_interface.stop_simulator()
    
    ego_agent_info = None
    if ego_agent:
        ego_agent_info = ego_agent.info
    
    return True, simulated_scenario, ego_vehicles, {
        'ego_agent_info': ego_agent_info,
        'plan_success_steps': plan_success_steps,
        'total_simu_steps': total_simu_steps,
        'list_of_time_costs': list_of_time_costs,
        'list_plan_visited_edges_count': list_plan_visited_edges_count,
        'list_of_obs_num': list_of_obs_num,
    }

def simulate_with_planner(scenario_index: int,
                          interactive_scenario_path: str,
                          motion_planner: BasicPlanner = None,
                          prediction_model: BasicPredictionModel = None,
                          max_num_of_steps: int = 100,
                          create_video: bool = False,
                          video_folder_path: str = None,
                          enable_debug_rviz_step: int = 1e+3,
                          random_seed: int= 0) \
        -> Tuple[bool, PlanningProblemSet, Dict[int, EgoVehicle], Scenario, Dict]:
    """
    Simulates an interactive scenario with a plugged in motion planner

    :param interactive_scenario_path: path to the interactive scenario folder
    :param motion_planner: planner class with port function plan()
    :param prediction_model: trained network-based prediction model
    :param max_num_of_steps: maximum number of simulation steps
    :param create_video: enable create video or not
    :param video_folder_path: path to store videos
    :param enable_debug_rviz_step: after which step, the debug rviz is enabled
    :return: Tuple of the sucess_flag, planning problem set, list of ego vehicles, simulated scenario, and dict_records
    """
    conf = load_sumo_configuration(interactive_scenario_path)
    conf.logging_level = 'INFO'

    # ['lcStrategic', 'lcSpeedGain', 'lcCooperative', 'sigma', 'speedDev', 'speedFactor', 'lcImpatience', 'impatience']
    # print(conf.driving_params.keys())
    # print(conf.driving_params.values())
    # print("\n\n\n")
    # conf.driving_params['impatience'] = '666' # code does not support this function!
    # conf.driving_params['jmTimegapMinor'] = 5.0
    # conf.driving_params['jmStoplineGap'] = 10.0

    scenario_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario

    preprocess_config = read_yaml_config(os.path.join(
        os.path.dirname(envs.__init__.__file__), 
        "conf/preprocess_simu_config.yaml"))['commonroad']
    print("preprocess_config parameter t_future=", preprocess_config['t_future'])

    # @note here use_sumo_manager == False, it is not tested when 
    #       prediction function is added.
    success, scenario_with_planner, ego_vehicles, dict_records =\
        simulate_with_prediction(conf,
                                 random_seed,
                                 preprocess_config,
                                 motion_planner,
                                 prediction_model,
                                 scenario_wrapper,
                                 scenario_index,
                                 interactive_scenario_path,
                                 num_of_steps=min(max_num_of_steps, conf.simulation_steps),
                                 planning_problem_set=planning_problem_set,
                                 use_sumo_manager=False,
                                 create_video=create_video,
                                 video_path=video_folder_path,
                                 enable_debug_rviz_step=enable_debug_rviz_step)
    
    if success == True:
        scenario_with_planner.scenario_id = scenario.scenario_id

    print("End simulation of {}.".format(conf.scenario_name))

    return success, planning_problem_set, ego_vehicles, scenario_with_planner, dict_records
