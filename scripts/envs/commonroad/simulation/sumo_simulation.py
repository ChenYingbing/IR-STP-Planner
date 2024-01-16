import copy
import logging
import math
import random
import sys
import time
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Union
from xml.etree import cElementTree as ET
import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.util import Interval
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State, Trajectory

import sumocr
from sumocr.interface.ego_vehicle import EgoVehicle
from sumocr.interface.util import *
from sumocr.maps.scenario_wrapper import AbstractScenarioWrapper
from sumocr.maps.sumo_scenario import ScenarioWrapper
# from sumocr.maps.util import add_params_in_rou_file
from sumocr.sumo_config import DefaultConfig, EGO_ID_START, ID_DICT
from sumocr.sumo_config.pathConfig import SUMO_BINARY, SUMO_GUI_BINARY

import libsumo as traci

__author__ = "Moritz Klischat, Maximilian Frühauf"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["ZIM Projekt ZF4086007BZ8"]
__version__ = "2022.1"
__maintainer__ = "Moritz Klischat"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"


class SumoSimulation:
    """
    Class for interfacing between the SUMO simulation and CommonRoad.
    """

    def __init__(self, set_random_seed: int = 0):
        """Init empty object"""
        self.set_random_seed = set_random_seed

        self.dt = None
        self.dt_sumo = None
        self.delta_steps = None
        self.planning_problem_set: PlanningProblemSet = None
        # {time_step: {obstacle_id: state}}
        self.obstacle_states: Dict[int, Dict[int, State]] = defaultdict(lambda: dict())
        self.simulationdomain = traci.simulation
        self.vehicledomain = traci.vehicle
        self.persondomain = traci.person
        self.routedomain = traci.route
        self._current_time_step = 0
        self.ids_sumo2cr, self.ids_cr2sumo = initialize_id_dicts(ID_DICT)
        self._max_lanelet_network_id = 0 # keep track of all IDs in CR lanelet_network

        # veh_id -> List[SignalState]
        self.signal_states: Dict[int, List[SignalState]] = defaultdict(list)
        self.obstacle_shapes: Dict[int, Rectangle] = dict()
        self.obstacle_types: Dict[int, ObstacleType] = dict()
        self.cached_position = {}  # caches position for orientation computation
        self._scenarios: ScenarioWrapper = None
        self.ego_vehicles: Dict[int, EgoVehicle] = dict()
        self.conf = DefaultConfig()
        self._silent = False
        # enables dummy synchronization of ego vehicles without planner for testing
        self.dummy_ego_simulation = False
        # ego sync parameters
        self._lc_duration_max = 10
        self._lc_counter = 0  # Count how many steps are counter for the lane change
        self._lc_inaction = 0  # Flag indicates that SUMO is performing a lane change for the ego
        self.lateral_position_buffer = dict()  # stores lateral position [ego_vehicle_id,float]
        self.logger: logging.Logger = None
        self._traci_label = None
        self.initialized = False

    @property
    def scenarios(self) -> ScenarioWrapper:
        return self._scenarios

    @scenarios.setter
    def scenarios(self, scenarios: ScenarioWrapper):
        def max_lanelet_network_id(lanelet_network: LaneletNetwork) -> int:
            max_lanelet = np.max([l.lanelet_id for l in lanelet_network.lanelets]) \
                if lanelet_network.lanelets else 0
            max_intersection = np.max([i.intersection_id for i in lanelet_network.intersections]) \
                if lanelet_network.intersections else 0
            max_traffic_light = np.max([t.traffic_light_id for t in lanelet_network.traffic_lights]) \
                if lanelet_network.traffic_lights else 0
            max_traffic_sign = np.max([t.traffic_sign_id for t in lanelet_network.traffic_signs]) \
                if lanelet_network.traffic_signs else 0
            val = np.max([max_lanelet, max_intersection, max_traffic_light, max_traffic_sign])
            if isinstance(val, np.generic):
                return val.item()
            else:
                return val

        if self.planning_problem_set is not None:
            max_pp= max(list(self.planning_problem_set.planning_problem_dict.keys()))
        else:
            max_pp = 0

        self._max_lanelet_network_id = max(max_pp,
                                           max_lanelet_network_id(scenarios.lanelet_network))
        self._scenarios = scenarios

    def initialize(self, conf: DefaultConfig,
                   scenario_wrapper: ScenarioWrapper,
                   planning_problem_set: PlanningProblemSet = None,
                   additional_file_path: str = None) -> None:
        """
        Reads scenario files, starts traci simulation, initializes vehicles, conducts pre-simulation.

        :param conf: configuration object. If None, use default configuration.
        :param scenario_wrapper: handles all files required for simulation. If None it is initialized with files
            folder conf.scenarios_path + conf.scenario_name
        :param planning_problem_set: initialize initial state of ego vehicles
        :param additional_file_path: additional file path for summo command
        (if None, use planning_problem_set from self.scenario)

        """
        if conf is not None:
            self.conf = conf

        self.logger = self._init_logging()

        assert isinstance(scenario_wrapper, AbstractScenarioWrapper), \
            f'scenario_wrapper expected type ScenarioWrapper or None, but got type {type(scenario_wrapper)}'
        self.scenarios = scenario_wrapper
        self.dt = self.conf.dt
        self.dt_sumo = self.conf.dt / self.conf.delta_steps
        self.delta_steps = self.conf.delta_steps
        self.planning_problem_set = planning_problem_set \
            if planning_problem_set is not None else self.scenarios.planning_problem_set

        assert sumocr.sumo_installed, "SUMO not installed or environment variable SUMO_HOME not set."

        if self.conf.with_sumo_gui:
            cmd = [
                SUMO_GUI_BINARY, "--start", "-c", self.scenarios.sumo_cfg_file,
                "--step-length",
                str(self.dt_sumo),
                "--lateral-resolution",
                str(self.conf.lateral_resolution)
            ]
        else:
            cmd = [
                SUMO_BINARY, "-c", self.scenarios.sumo_cfg_file,
                "--step-length",
                str(self.dt_sumo),
                "--lateral-resolution",
                str(self.conf.lateral_resolution)
            ]

        if self.conf.lateral_resolution > 0.0:
            cmd.extend(['--lanechange.duration', '0'])

        # if self.conf.random_seed:
        #     np.random.seed(self.conf.random_seed)
        #     random.seed(self.conf.random_seed)
        #     cmd.extend(['--seed', str(self.conf.random_seed)])
        cmd.extend(['--seed', str(self.set_random_seed)])

        # if additional_file_path:
        cmd.extend(['--default.action-step-length', str(0.1)])

        traci.start(cmd)
        # # [modified]
        # traci.vehicletype.setParameter("passenger", "carFollowModel", "IDM")
        for tid in traci.trafficlight.getIDList():
            traci.trafficlight.setProgram(tid, None)

        # simulate until ego_time_start
        if self.planning_problem_set is not None and len(self._find_ego_vehicles_in_rou_file()) == 0:
            self.__presimulation_silent(max(0, self.conf.presimulation_steps-1), fetch_obstacles=False)
            if len(self.ego_vehicles) > 0:
                self.logger.warning('<SumoSimulation/init_ego_vehicles> Ego vehicles are already defined through .rou'
                                    'file and planning problem!')
            self.initialized = True
            self.init_ego_vehicles_from_planning_problem(self.planning_problem_set)
            self.__presimulation_silent(1, fetch_obstacles=True)
        else:
            self.__presimulation_silent(self.conf.presimulation_steps, fetch_obstacles=True)
            self.initialized = True

    def _init_logging(self):
        # Create a custom logger
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(level=getattr(logging, self.conf.logging_level))
        if not logger.hasHandlers():
            # Create handlers
            c_handler = logging.StreamHandler()

            # Create formatters and add it to handlers
            c_format = logging.Formatter('<%(name)s.%(funcName)s> %(message)s')
            c_handler.setFormatter(c_format)

            # Add handlers to the logger
            logger.addHandler(c_handler)

        return logger

    def _find_ego_vehicles_in_rou_file(self, n_max=1) -> List[str]:
        rou_file = self.scenarios.get_rou_file()
        if rou_file is None:
            return []
        with open(rou_file, 'r') as f:
            tree = ET.parse(f)

        ego_vehicle_ids = []
        vehicles = tree.findall("vehicle")
        for v in vehicles:
            if v.get("id").startswith(EGO_ID_START):
                ego_vehicle_ids.append(v.get("id"))
                if len(ego_vehicle_ids) >= n_max:
                    break

        return ego_vehicle_ids

    def init_ego_vehicles_from_planning_problem(
            self, planning_problem_set: PlanningProblemSet) -> None:
        """
        Initializes the ego vehicles according to planning problem set.

        :param planning_problem_set: The planning problem set which defines the ego vehicles.
        """
        assert self.initialized, 'Initialize the SumoSimulation first!'

        self.dummy_ego_simulation = False

        # retrieve arbitrary route id for initialization (will not be used by interface)
        generic_route_id = traci.route.getIDList()[0]

        width = self.conf.ego_veh_width
        length = self.conf.ego_veh_length

        # create ego vehicles if planning problem is given
        if planning_problem_set is not None:
            list_ids_ego_used = []
            for planning_problem in planning_problem_set.planning_problem_dict.values():
                new_id = self._create_sumo_id(list_ids_ego_used)
                list_ids_ego_used.append(new_id)
                sumo_id = EGO_ID_START + str(new_id)
                cr_id = self._create_cr_id(type='egoVehicle', sumo_id=sumo_id, sumo_prefix=EGO_ID_START)
                self.vehicledomain.add(sumo_id, generic_route_id, typeID="DEFAULT_VEHTYPE",
                                       depart="now",
                                       departLane='first', departPos="base",
                                       departSpeed=planning_problem.initial_state.velocity,
                                       arrivalLane="current", arrivalPos="max",
                                       arrivalSpeed="current",
                                       fromTaz="", toTaz="", line="", personCapacity=0,
                                       personNumber=0)
                sumo_angle = 90.0 - math.degrees(planning_problem.initial_state.orientation)

                position_tmp = planning_problem.initial_state.position\
                               + 0.5 * length * np.array([math.cos(planning_problem.initial_state.orientation),
                                                          math.sin(planning_problem.initial_state.orientation)])
                self.vehicledomain.setLength(sumo_id, length=length)
                self.vehicledomain.setWidth(sumo_id, width=width)
                self.vehicledomain.moveToXY(sumo_id, edgeID='dummy', laneIndex=-1,
                                            x=position_tmp[0],
                                            y=position_tmp[1],
                                            angle=sumo_angle, keepRoute=2)

                self.ids_sumo2cr[EGO_ID_START][sumo_id] = cr_id
                self.ids_sumo2cr['vehicle'][sumo_id] = cr_id
                self.ids_cr2sumo[EGO_ID_START][cr_id] = sumo_id
                self.ids_cr2sumo['vehicle'][cr_id] = sumo_id
                self._add_ego_vehicle(EgoVehicle(cr_id, planning_problem.initial_state,
                                                 self.conf.delta_steps, width, length,
                                                 planning_problem))

    def _add_ego_vehicle(self, ego_vehicle: EgoVehicle):
        """
        Adds a ego vehicle to the current ego vehicle set.

        :param ego_vehicle: the ego vehicle to be added.
        """
        if ego_vehicle.id in self._ego_vehicles:
            self.logger.debug(
                f'Ego vehicle with id {ego_vehicle.id} already exists!',
                exc_info=True)
        else:
            self._ego_vehicles[ego_vehicle.id] = ego_vehicle

    @property
    def ego_vehicles(self) -> Dict[int, EgoVehicle]:
        """
        Returns the ego vehicles of the current simulation.

        """
        return self._ego_vehicles

    @ego_vehicles.setter
    def ego_vehicles(self, ego_vehicles: Dict[int, EgoVehicle]):
        """
        Sets the ego vehicles of the current simulation.

        :param ego_vehicles: ego vehicles used to set up the current simulation.
        """
        self._ego_vehicles = ego_vehicles

    @property
    def current_time_step(self) -> int:
        """
        :return: current time step of interface
        """
        return self._current_time_step

    @current_time_step.setter
    def current_time_step(self, current_time_step):
        """
        Time step should not be set manually.
        """
        raise (ValueError('Time step should not be set manually'))

    def commonroad_scenario_at_time_step(self, time_step: int, add_ego=False, start_0=True) -> Scenario:
        """
        Creates and returns a commonroad scenario at the given time_step. Initial time_step=0 for all obstacles.

        :param time_step: the scenario will be created according this time step.
        :param add_ego: whether to add ego vehicles to the scenario.
        :param start_0: if set to true, initial time step of vehicles is 0, otherwise, the current time step


        """
        self.cr_scenario = self.scenarios.create_minimal_scenario()

        # remove old obstacles from lanes
        # this is only necessary if obstacles are added to lanelets
        if self.conf.add_lanelets_to_dyn_obstacles:
            for lanelet in self.cr_scenario.lanelet_network.lanelets:
                lanelet.dynamic_obstacles_on_lanelet = {}

        self.cr_scenario.add_objects(self._get_cr_obstacles_at_time(time_step, add_ego=add_ego, start_0=start_0))
        return self.cr_scenario

    def commonroad_scenarios_all_time_steps(self) -> Scenario:
        """
        Creates and returns a commonroad scenario with all the dynamic obstacles.
        :param lanelet_network:
        :return: list of cr scenarios, list of cr scenarios with ego, list of planning problem sets)
        """
        self.cr_scenario = self.scenarios.create_full_meta_scenario()
        # remove old obstacles from lanes
        # this is only necessary if obstacles are added to lanelets
        if self.conf.add_lanelets_to_dyn_obstacles:
            for lanelet in self.cr_scenario.lanelet_network.lanelets:
                lanelet.dynamic_obstacles_on_lanelet = {}

        self.cr_scenario.add_objects(self._get_cr_obstacles_all())
        return self.cr_scenario

    def _init_vehicle_ids(self) -> Tuple[list, list]:
        """
        Initialize vehicle IDs and set individual vehicle parameters.
        :return lists of new vehicle and pedestrian IDs that entered the simulation
        """
        new_ids = []
        vehicle_ids = self.vehicledomain.getIDList()
        for veh_id in vehicle_ids:
            if veh_id not in self.ids_sumo2cr[SUMO_VEHICLE_PREFIX]:
                if veh_id.startswith(EGO_ID_START):
                    obs_type = EGO_ID_START
                else:
                    obs_type = 'obstacleVehicle'

                self._create_cr_id(obs_type, veh_id, SUMO_VEHICLE_PREFIX)
                vehicle_class = VEHICLE_TYPE_SUMO2CR[self.vehicledomain.getVehicleClass(str(veh_id))]
                self._set_veh_params(self.vehicledomain, veh_id, vehicle_class)

                # [modified]
                # vehicle_type_str = self.vehicledomain.getVehicleClass(str(veh_id))
                # if (vehicle_type_str == 'truck') or (vehicle_type_str == 'bus'):
                #     # truck is too long, make it shorter
                #     vehicle_class = VEHICLE_TYPE_SUMO2CR['passenger']
                #     self._set_veh_params(self.vehicledomain, veh_id, vehicle_class)

                # [modified]
                # traci.vehicle.setParameter(veh_id, "carFollowModel", "IDM")
                traci.vehicle.setMinGap(veh_id, 3.0) # default is 2.5 m
                # traci.vehicle.setImperfection(veh_id, 0.0)
                # traci.vehicle.setParameter(veh_id, "sigma", "0.0")
                # traci.vehicle.setParameter(veh_id, "jmSigmaMinor", "0.0")
                # traci.vehicle.setTau(veh_id, 1.5) # desired time headway (seconds)
                # traci.vehicle.setParameter(veh_id, "s0", "2.0") # minimum gap of distance (meters)

                new_ids.append(veh_id)

        person_ids = self.persondomain.getIDList()
        obstacle_type = "obstaclePedestrian"
        new_ped_ids = []
        for ped_id in person_ids:
            if ped_id not in self.ids_sumo2cr[SUMO_PEDESTRIAN_PREFIX]:
                # initialize new pedestrian
                self._create_cr_id(obstacle_type, ped_id, SUMO_PEDESTRIAN_PREFIX)
                vehicle_class = VEHICLE_TYPE_SUMO2CR[self.persondomain.getTypeID(ped_id).split("@")[0]]
                self._set_veh_params(self.persondomain, ped_id, vehicle_class)
                new_ped_ids.append(ped_id)

        return new_ids, new_ped_ids

    def simulate_step(self) -> None:
        """
        Executes next simulation step (consisting of delta_steps sub-steps with dt_sumo=dt/delta_steps) in SUMO

        """

        # simulate sumo scenario for delta_steps time steps
        for i in range(self.delta_steps):
            # send ego vehicles to SUMO
            if not self.dummy_ego_simulation and len(self.ego_vehicles) > 0:
                self._send_ego_vehicles(self.ego_vehicles, i)

            # execute SUMO simulation step
            traci.simulationStep()
            for ego_veh in list(self.ego_vehicles.values()):
                ego_veh._current_time_step += 1

        # get updated obstacles from sumo
        self._current_time_step += 1
        new_vehicle_ids, new_ped_ids = self._init_vehicle_ids()
        self._fetch_sumo_vehicles(self.current_time_step, new_vehicle_ids)
        self._fetch_sumo_pedestrians(self.current_time_step, new_ped_ids)

    def __presimulation_silent(self, pre_simulation_steps: int, fetch_obstacles=True):
        """
        Simulate SUMO without synchronization of interface. Used before starting interface simulation.

        :param pre_simulation_steps: the steps of simulation which are executed before checking the existence
        of ego vehicles and configured simulation step.
        :param fetch_obstacles: fetch obstacles after simulating

        """
        assert self.current_time_step == 0
        assert pre_simulation_steps >= 0, f'ego_time_start={self.conf.presimulation_steps} must be >0'

        if pre_simulation_steps == 0:
            return

        self._silent = True
        for i in range(pre_simulation_steps * self.delta_steps):
            traci.simulationStep()
            self._init_vehicle_ids()

        self._silent = False
        if fetch_obstacles:
            self._fetch_sumo_vehicles(self.current_time_step, self.vehicledomain.getIDList())
            self._fetch_sumo_pedestrians(self.current_time_step, self.persondomain.getIDList())

    def _subscribe_new(self, new_ids: List[str], domain=traci.vehicle):
        """
        Subscribe to values of new vehicle/pedestrian
        :param new_ids: list of new SUMO IDs
        :param domain: domain from which should be subscribed
        :return:
        """
        vehicle_ids = self.sort_ego_first(new_ids)
        for veh_id in vehicle_ids:
            domain.subscribe(veh_id, traci_subscription_values)

    def _get_subscription_results(self, domain=traci.vehicle) -> Dict[str, Dict[int, any]]:
        """
        Return subscription results from current time step
        :param domain:
        :return: dict with results with structure {sumo_id: {value_id: value}}
        """
        return domain.getAllSubscriptionResults()

    def _fetch_sumo_vehicles(self, time_step: int, new_ids: List[str]):
        """
        Gets and stores all vehicle states from SUMO. Initializes ego vehicles when they enter simulation.

        """
        self._subscribe_new(new_ids)
        vehicle_states = self._get_subscription_results()

        if len(vehicle_states) == 0:
            return

        if not self.dummy_ego_simulation:
            self.check_ego_collisions(raise_error=True)

        for veh_id, vehicle_state in vehicle_states.items():
            state = self._extract_current_state(self.vehicledomain, veh_id, vehicle_state,
                                                SUMO_VEHICLE_PREFIX)
            if state is None:
                continue

            # initializes new vehicle
            if veh_id in new_ids:
                if veh_id.startswith(EGO_ID_START) and not self._silent:
                    # new ego vehicle
                    cr_id = self.ids_sumo2cr[EGO_ID_START][veh_id]
                    if self.dummy_ego_simulation:
                        state.time_step = time_step - 1
                    else:
                        state.time_step = time_step

                    if self.planning_problem_set is not None:
                        planning_problem = list(self.planning_problem_set.planning_problem_dict.values())[0]
                    elif self.scenarios.planning_problem_set is not None:
                        planning_problem = list(self.scenarios.planning_problem_set.planning_problem_dict.values())[0]
                    else:
                        planning_problem = None

                    self._add_ego_vehicle(EgoVehicle(cr_id, state, self.conf.delta_steps,
                                                     self.conf.ego_veh_width,
                                                     self.conf.ego_veh_length,
                                                     planning_problem=planning_problem))
                elif not self._silent:
                    # new obstacle vehicle
                    cr_id = self.ids_sumo2cr['obstacleVehicle'][veh_id]
                    vehicle_class = VEHICLE_TYPE_SUMO2CR[self.vehicledomain.getVehicleClass(str(veh_id))]
                    self.obstacle_types[cr_id] = vehicle_class
                    self.obstacle_shapes[cr_id] = self._get_veh_shape(self.vehicledomain, veh_id)
                    self.obstacle_states[time_step][self.ids_sumo2cr['obstacleVehicle'][veh_id]] = state
            elif veh_id in self.ids_sumo2cr['obstacleVehicle']:
                # get obstacle vehicle state
                self.obstacle_states[time_step][self.ids_sumo2cr['obstacleVehicle'][veh_id]] = state
            elif not self._silent and veh_id not in self.ids_sumo2cr['egoVehicle']:
                raise NotImplemented()

            # read signal state
            if not self._silent:
                signal_states = get_signal_state(vehicle_state[traci.constants.VAR_SIGNALS], self.current_time_step)
                key = self.ids_sumo2cr['obstacleVehicle'][veh_id] \
                    if veh_id in self.ids_sumo2cr['obstacleVehicle'] else self.ids_sumo2cr[EGO_ID_START][veh_id]
                self.signal_states[key].append(signal_states)

            """For testing with dummy_ego_simulation"""
            if not self._silent and self.dummy_ego_simulation and veh_id in self.ids_sumo2cr['egoVehicle']:
                ego_veh = self.ego_vehicles[self.ids_sumo2cr['egoVehicle'][veh_id]]
                ori = state.orientation
                state_list = []
                for t in range(0, self.conf.delta_steps):
                    state_tmp = copy.deepcopy(state)
                    state_tmp.position = state.position + (t + 1) * state.velocity \
                                         * self.dt * np.array([math.cos(ori), math.sin(ori)])
                    state_tmp.time_step = t + 1
                    state_list.append(state_tmp)

                ego_veh.set_planned_trajectory(state_list)

    def _fetch_sumo_pedestrians(self, time_step: int, new_ids: List[str]):
        """
        Gets and stores all vehicle states from SUMO. Initializes ego vehicles when they enter simulation.

        """
        self._subscribe_new(new_ids, domain=traci.person)
        person_states = self._get_subscription_results(domain=traci.person)

        if not person_states:
            return

        obstacle_type = "obstaclePedestrian"
        for ped_id, ped_state in person_states.items():
            state = self._extract_current_state(self.persondomain, ped_id, ped_state, SUMO_PEDESTRIAN_PREFIX)
            if state is None:
                continue
            if ped_id in new_ids:
                # initialize new pedestrian
                cr_id = self._create_cr_id(obstacle_type, ped_id, SUMO_PEDESTRIAN_PREFIX)
                vehicle_class = VEHICLE_TYPE_SUMO2CR[self.persondomain.getTypeID(ped_id).split("@")[0]]
                shape = self._set_veh_params(self.persondomain, ped_id, vehicle_class)

                self.obstacle_types[cr_id] = vehicle_class
                self.obstacle_shapes[cr_id] = shape
                self.obstacle_states[time_step][self.ids_sumo2cr[obstacle_type][ped_id]] = state
            elif ped_id in self.ids_sumo2cr[obstacle_type]:
                # get obstacle vehicle state
                self.obstacle_states[time_step][self.ids_sumo2cr[obstacle_type][ped_id]] = state

    def _get_veh_shape(self, vehicledomain, sumo_id):
        return Rectangle(vehicledomain.getLength(sumo_id), vehicledomain.getWidth(sumo_id))

    def check_ego_collisions(self, raise_error=True):
        """
        Checks if the ego vehicle is colliding in the current time step of the simulation.
        :param raise_error: raise EgoCollisionError, other return False
        :return: False if collision occurs, True otherwise
        """
        colliding_vehicle_ids = self.simulationdomain.getCollidingVehiclesIDList()
        for id_ in colliding_vehicle_ids:
            if EGO_ID_START in id_:
                if raise_error:
                    raise EgoCollisionError(self.current_time_step)
                else:
                    return False

        return True

    def _set_veh_params(self, domain,
                        veh_id: int,
                        vehicle_class: ObstacleType) -> Rectangle:
        """

        :param domain: traci domain for which the parameter should be set
        :type domain: Union[PersonDomain, VehicleDomain]
        :param veh_id: vehicle ID
        :param vehicle_class: vehicle class
        :return:
        """
        def sample(val: Union[Interval, float]) -> float:
            if isinstance(val, Interval):
                assert 0 <= val.start <= val.end, f"All values in the interval need to be positive: {val}"
                return float(np.random.uniform(val.start, val.end))
            else:
                return val

        try:
            width = sample(self.conf.veh_params["width"][vehicle_class])
            length = sample(self.conf.veh_params["length"][vehicle_class])
            min_gap = sample(self.conf.veh_params["minGap"][vehicle_class])
            domain.setWidth(veh_id, width)
            domain.setLength(veh_id, length)
            domain.setMinGap(veh_id, min_gap)
        except KeyError as e:
            self.logger.warning(f"vehicle_class: {vehicle_class} is invalid")
            raise e
        try:
            accel = sample(self.conf.veh_params["accel"][vehicle_class])
            decel = sample(self.conf.veh_params["decel"][vehicle_class])
            max_speed = sample(self.conf.veh_params["maxSpeed"][vehicle_class])
            domain.setAccel(veh_id, accel)
            domain.setDecel(veh_id, decel)
            domain.setMaxSpeed(veh_id, max_speed)
        # in case of a pedestrian domain
        finally:
            return Rectangle(length, width)

    def _extract_current_state(self, domain, veh_id: str, vehicle_state: Dict[int, any],
                               sumo_prefix: str) -> Union[None, State]:
        """
        Gets the current state from sumo.
        :param domain
        :type: Union[PersonDomain, VehicleDomain]
        :param veh_id: the id of the vehicle, whose state will be returned from SUMO.

        :return: the state of the given vehicle or None, if not in field of view
        """
        unique_id = sumo_prefix + veh_id
        position = np.array(vehicle_state[traci.constants.VAR_POSITION])
        velocity = math.sqrt(vehicle_state[traci.constants.VAR_SPEED]**2 + vehicle_state[traci.constants.VAR_SPEED_LAT]**2)

        cr_id = sumo2cr(veh_id, self.ids_sumo2cr)
        if cr_id and cr_id in self.obstacle_shapes:
            length = self.obstacle_shapes[cr_id].length
        else:
            length = domain.getLength(veh_id)

        if self.conf.compute_orientation \
                and self.current_time_step > 1 \
                and velocity > 0.5 \
                and unique_id in self.cached_position:
            delta_pos = position - self.cached_position[unique_id]
            orientation = math.atan2(delta_pos[1], delta_pos[0])
        else:
            orientation = math.radians(-vehicle_state[traci.constants.VAR_ANGLE] + 90.0)

        self.cached_position[unique_id] = position
        position -= 0.5 * length * np.array([math.cos(orientation), math.sin(orientation)])

        in_fov = self.is_in_field_of_view(position, veh_id)
        if in_fov is False:
            return None

        acceleration = vehicle_state[traci.constants.VAR_ACCELERATION]

        return State(position=position,
                     orientation=orientation,
                     velocity=velocity,
                     acceleration=acceleration,
                     time_step=self.current_time_step)

    def _get_cr_obstacles_at_time(self, time_step: int,
                                  add_ego: bool = False,
                                  start_0: bool = False) -> List[DynamicObstacle]:
        """
        Gets current state of all vehicles in commonroad format from recorded simulation.

        :param time_step: time step of scenario
        :param add_ego: if True, add ego vehicles as well
        :param start_0: if True, initial time step of vehicles is 0, otherwise, the current time step

        """

        vehicle_dict: Dict[int, State] = self.obstacle_states[time_step]
        obstacles: List[DynamicObstacle] = []

        for veh_id, state in vehicle_dict.items():
            if start_0:
                state.time_step = 0
            else:
                state.time_step = time_step

            center_lanelets = None
            shape_lanelets = None
            if self.conf.add_lanelets_to_dyn_obstacles:
                center_lanelets = {lanelet_id
                                   for lanelet_ids in
                                   self.scenarios.lanelet_network.find_lanelet_by_position([state.position])
                                   for lanelet_id in lanelet_ids}
                shape_lanelets = {lanelet_id
                                  for lanelet_ids in
                                  self.scenarios.lanelet_network.find_lanelet_by_position(
                                      [state.position + v for v in self.obstacle_shapes[veh_id].vertices])
                                  for lanelet_id in lanelet_ids}

            obstacle_type = self.obstacle_types[veh_id]
            signal_state = next((s for s in self.signal_states[veh_id] if s.time_step == time_step), None)
            if start_0:
                signal_state.time_step = 0
            else:
                signal_state.time_step = time_step

            dynamic_obstacle = DynamicObstacle(
                obstacle_id=veh_id,
                obstacle_type=obstacle_type,
                initial_state=state,
                obstacle_shape=self.obstacle_shapes[veh_id],
                initial_center_lanelet_ids=center_lanelets if center_lanelets else None,
                initial_shape_lanelet_ids=shape_lanelets if shape_lanelets else None,
                initial_signal_state=signal_state)
            obstacles.append(dynamic_obstacle)

        if add_ego:
            obstacles.extend(self.get_ego_obstacles(time_step))
        return obstacles

    def _get_cr_obstacles_all(self) -> List[DynamicObstacle]:
        """
        For all recorded time steps, get states of all obstacles and convert them into commonroad dynamic obstacles. :return: list of dynamic obstacles
        """
        # transform self.obstacle_states:Dict[time_step:[veh_id:State]] to veh_state:Dict[veh_id:[time_step:State]]
        veh_state = {}
        for time_step, veh_dicts in self.obstacle_states.items():
            for veh_id, state in veh_dicts.items():
                veh_state[veh_id] = {}

        for time_step, veh_dicts in self.obstacle_states.items():
            for veh_id, state in veh_dicts.items():
                state.time_step = time_step
                veh_state[veh_id][time_step] = state

        # get all vehicles' ids for id conflict check between lanelet_id and veh_id
        self.veh_ids: List = [*veh_state]

        # create cr obstacles
        obstacles = []
        for veh_id, time_dict in veh_state.items():
            state_list = list(time_dict.values())
            # coordinate transformation for all positions from sumo format to commonroad format
            obstacle_shape = self.obstacle_shapes[veh_id]
            obstacle_type = self.obstacle_types[veh_id]
            initial_state = state_list[0]

            assert self.conf.video_start != self.conf.video_end, \
                "Simulation start time and end time are the same. Please set simulation interval."

            if len(state_list) > 4:
                obstacle_trajectory = Trajectory(state_list[0].time_step, state_list[0:])
                obstacle_prediction = TrajectoryPrediction(obstacle_trajectory, obstacle_shape)
                center_lanelets = None
                shape_lanelets = None
                if self.conf.add_lanelets_to_dyn_obstacles:
                    center_lanelets = {lanelet_id for lanelet_ids in
                                       self.cr_scenario.lanelet_network.find_lanelet_by_position(
                                           [initial_state.position])
                                       for lanelet_id in lanelet_ids}
                    shape_lanelets = {lanelet_id for lanelet_ids in
                                      self.cr_scenario.lanelet_network.find_lanelet_by_position(
                                          [initial_state.position + v
                                           for v in self.obstacle_shapes[veh_id].vertices])
                                      for lanelet_id in lanelet_ids}

                signal_states = self.signal_states[veh_id]
                dynamic_obstacle = DynamicObstacle(
                    obstacle_id=veh_id,
                    obstacle_type=obstacle_type,
                    initial_state=initial_state,
                    obstacle_shape=obstacle_shape,
                    prediction=obstacle_prediction,
                    initial_center_lanelet_ids=center_lanelets if center_lanelets else None,
                    initial_shape_lanelet_ids=shape_lanelets if shape_lanelets else None,
                    initial_signal_state=signal_states[0] if signal_states else None,
                    signal_series=signal_states[1:] if signal_states else None)  # add a trajectory element
                obstacles.append(dynamic_obstacle)
            else:
                self.logger.debug(
                    f'Vehicle {veh_id} has been simulated less than 5 time steps. Not converted to cr obstacle.'
                )
        return obstacles

    @lru_cache()
    def _get_ids_of_map(self) -> Set[int]:
        """
        Get a list of ids of all the lanelets from the cr map which is converted from a osm map.
        :return: list of lanelets' ids
        """
        ids = set()
        for lanelet in self.scenarios.lanelet_network.lanelets:
            ids.add(lanelet.lanelet_id)
        for ts in self.scenarios.lanelet_network.traffic_signs:
            ids.add(ts.traffic_sign_id)
        for ts in self.scenarios.lanelet_network.traffic_lights:
            ids.add(ts.traffic_light_id)
        for ts in self.scenarios.lanelet_network.intersections:
            ids.add(ts.intersection_id)
            for inc in ts.incomings:
                ids.add(inc.incoming_id)
        return ids

    def get_ego_obstacles(self, time_step: Union[int, None] = None) -> List[DynamicObstacle]:
        """
        Get list of ego vehicles converted to Dynamic obstacles
        :param time_step: initial time step, if None, get complete driven trajectory
        :return:
        """
        obstacles = []
        for veh_id, ego_veh in self.ego_vehicles.items():
            obs = ego_veh.get_dynamic_obstacle(time_step)
            if obs is not None:
                obstacles.append(obs)

        return obstacles

    def _send_ego_vehicles(self, ego_vehicles: Dict[int, EgoVehicle], delta_step: int = 0) -> None:
        """
        Sends the information of ego vehicles to SUMO.

        :param ego_vehicles: list of dictionaries.
            For each ego_vehicle, write tuple (cr_ego_id, cr_position, cr_lanelet_id, cr_orientation, cr_lanelet_id)
            cr_lanelet_id can be omitted but this is not recommended, if the lanelet is known for sure.
        :param delta_step: which time step of the planned trajectory should be sent

        """
        for id_cr, ego_vehicle in ego_vehicles.items():
            assert ego_vehicle.current_time_step == self.current_time_step, \
                f'Trajectory of ego vehicle has not been updated. Still at time_step {ego_vehicle.current_time_step},' \
                f'while simulation step {self.current_time_step + 1} should be simulated.'

            planned_state = ego_vehicle.get_planned_state(delta_step)
            self._forward_info2sumo(planned_state, "SYNC_MOVE_XY", 0, id_cr)

    def _get_ego_ids(self) -> Dict[int, str]:
        """
        Returns a dictionary with all current ego vehicle ids and corresponding sumo ids
        """
        return self.ids_cr2sumo[EGO_ID_START]

    def _create_sumo_id(self, list_ids_ego_used: List[int]) -> int:
        """
        Generates a new unused id for SUMO
        :return:
        """
        id_list = traci.vehicle.getIDList()
        new_id = int(len(id_list))
        i = 0
        while i < 1000:
            if str(new_id) not in id_list and new_id not in list_ids_ego_used:
                return new_id
            else:
                new_id += 1
                i += 1

    def _create_cr_id(self, type: str, sumo_id: str, sumo_prefix: str, cr_id: int = None) -> int:
        """
        Generates a new cr ID and adds it to ID dictionaries

        :param type: one of the keys in params.id_convention; the type defines the first digit of the cr_id
        :param sumo_id: id in sumo simulation
        :param sumo_prefix: str giving what set of sumo ids to use

        :return: cr_id as int
        """
        cr_id = generate_cr_id(type, sumo_id, sumo_prefix, self.ids_sumo2cr, self._max_lanelet_network_id)

        self.ids_sumo2cr[type][sumo_id] = cr_id
        self.ids_sumo2cr[sumo_prefix][sumo_id] = cr_id
        self.ids_cr2sumo[type][cr_id] = sumo_id
        self.ids_cr2sumo[sumo_prefix][cr_id] = sumo_id
        return cr_id

    @property
    def _silent(self):
        """Ego vehicle is not synced in this mode."""
        return self.__silent

    @_silent.setter
    def _silent(self, silent):
        assert self.current_time_step == 0
        self.__silent = silent

    def stop(self):
        """ Exits SUMO Simulation"""
        traci.close()
        sys.stdout.flush()

    # Ego sync functions
    def check_lanelets_future_change(
            self, current_state: State,
            planned_traj: List[State]) -> Tuple[str, int]:
        """
        Checks the lanelet changes of the ego vehicle in the future time_window.

        :param lanelet_network: object of the lanelet network
        :param time_window: the time of the window to check the lanelet change
        :param traj_index: index of the planner output corresponding to the current time step

        :return: lc_status, lc_duration: lc_status is the status of the lanelet change in the next time_window; lc_duration is the unit of time steps (using sumo dt)

        """
        lc_duration_max = min(self.conf.lanelet_check_time_window,
                              len(planned_traj))
        lanelet_network = self.scenarios.lanelet_network
        lc_status = 'NO_LC'
        lc_duration = 0

        # find current lanelets
        current_position = current_state.position
        current_lanelets_ids = lanelet_network.find_lanelet_by_position([current_position])[0]
        current_lanelets = [
            lanelet_network.find_lanelet_by_id(id)
            for id in current_lanelets_ids
        ]

        # check for lane change
        for current_lanelet in current_lanelets:
            for t in range(lc_duration_max):
                future_lanelet_ids = lanelet_network.find_lanelet_by_position(
                    [planned_traj[t].position])[0]
                if current_lanelet.adj_right in future_lanelet_ids:
                    lc_status = 'RIGHT_LC'
                    lc_duration = 2 * t * self.conf.delta_steps
                    break
                elif current_lanelet.adj_left in future_lanelet_ids:
                    lc_status = 'LEFT_LC'
                    lc_duration = 2 * t * self.conf.delta_steps
                    break
                else:
                    pass

        self.logger.debug('current lanelets: ' + str(current_lanelets))
        self.logger.debug('lc_status=' + lc_status)
        self.logger.debug('lc_duration=' + str(lc_duration))
        return lc_status, lc_duration

    def _check_lc_start(self, ego_id: str, lc_future_status: str) -> str:
        """
        This function checks if a lane change is started according to the change in the lateral position and the lanelet
        change prediction. Note that checking the change of lateral position only is sensitive to the tiny changes, also
        at the boundaries of the lanes the lateral position sign is changed because it will be calculated relative to
        the new lane. So we check the future lanelet change to avoid these issues.

        :param ego_id: id of the ego vehicle
        :param lc_future_status: status of the future lanelet changes

        :return: lc_status: the status whether the ego vehicle starts a lane change or no
        """
        lateral_position = self.vehicledomain.getLateralLanePosition(
            cr2sumo(ego_id, self.ids_cr2sumo))

        if lc_future_status == 'NO_LC' or not id in self.lateral_position_buffer:
            lc_status = 'NO_LC'
        elif lc_future_status == 'RIGHT_LC' \
            and self.lateral_position_buffer[id] > self.conf.lane_change_tol + lateral_position:
            lc_status = 'RIGHT_LC_STARTED'
        elif lc_future_status == 'LEFT_LC' \
            and self.lateral_position_buffer[id] < -self.conf.lane_change_tol + lateral_position:
            lc_status = 'LEFT_LC_STARTED'
        else:
            lc_status = 'NO_LC'

        self.logger.debug('LC current status: ' + lc_status)

        self.lateral_position_buffer[id] = lateral_position

        return lc_status

    def _consistency_protection(self, ego_id: str,
                                current_state: State) -> str:
        """
        Checks the L2 distance between SUMO position and the planner position and returns CONSISTENCY_ERROR if it is
        above the configured margin.

        :param ego_id: id of the ego vehicle (string)
        :param current_state: the current state read from the commonroad motion planner

        :return: retval: the status whether there is a consistency error between sumo and planner positions or not
        """
        cons_error = 'CONSISTENCY_NO_ERROR'

        pos_sumo = self.vehicledomain.getPosition(cr2sumo(ego_id, self.ids_cr2sumo))
        pos_cr = current_state.position
        dist_error = np.linalg.norm(pos_cr - pos_sumo)
        if dist_error > self.conf.protection_margin:
            cons_error = 'CONSISTENCY_ERROR'

        self.logger.debug('SUMO X: ' + str(pos_sumo[0]) + ' **** SUMO Y: ' + str(pos_sumo[1]))
        self.logger.debug('TRAJ X: ' + str(pos_cr[0]) + ' **** TRAJ Y: ' + str(pos_cr[1]))
        self.logger.debug('Error Value: ' + str(dist_error))
        self.logger.debug('Error Status: ' + cons_error)

        if self._lc_inaction == 0:
            cons_error = 'CONSISTENCY_NO_ERROR'
        return cons_error

    def _check_sync_mechanism(self, lc_status: str, ego_id: int,
                              current_state: State) -> str:
        """
        Defines the sync mechanism type that should be executed according to the ego vehicle motion.

        :param lc_status: status of the lanelet change in the next time_window
        :param ego_id: id of the ego vehicle (string)
        :param current_state: the current state read from the commonroad motion planner

        :return: retval: the sync mechanism that should be followed while communicating from the interface to sumo
        """
        if self.conf.lane_change_sync == True:
            # Check the error between SUMO and CR positions
            cons_error = self._consistency_protection(ego_id, current_state)
            if cons_error == 'CONSISTENCY_NO_ERROR':  # CONSISTENCY_NO_ERROR means error below the configured margin
                if self._lc_inaction == 0:
                    if lc_status == 'RIGHT_LC_STARTED':
                        self._lc_inaction = 1
                        retval = 'SYNC_SUMO_R_LC'
                    elif lc_status == 'LEFT_LC_STARTED':
                        self._lc_inaction = 1
                        retval = 'SYNC_SUMO_L_LC'
                    else:
                        retval = 'SYNC_MOVE_XY'
                else:  # There is a lane change currently in action so do nothing and just increament the counter
                    self._lc_counter += 1
                    if self._lc_counter >= self._lc_duration_max:
                        self._lc_counter = 0
                        self._lc_inaction = 0
                    retval = 'SYNC_DO_NOTHING'
            else:  # There is a consistency error so force the sync mechanism to moveToXY to return back to zero error
                retval = 'SYNC_MOVE_XY'
                self._lc_counter = 0
                self._lc_inaction = 0
        else:
            retval = 'SYNC_MOVE_XY'

        self.logger.debug('Sync Mechanism is: ' + retval)
        self.logger.debug('Lane change performed since ' +
                          str(self._lc_counter))
        return retval

    def _forward_info2sumo(self, planned_state: State, sync_mechanism: str, lc_duration: int, ego_id: int):
        """
        Forwards the information to sumo (either initiate moveToXY or changeLane) according to the sync mechanism.

        :param planned_state: the planned state from commonroad motion planner
        :param sync_mechanism: the sync mechanism that should be followed while communicating from the interface to sumo
        :param lc_duration: lane change duration, expressed in number of time steps
        :param ego_id: id of the ego vehicle
        """
        id_sumo = cr2sumo(ego_id, self.ids_cr2sumo)

        if sync_mechanism == 'SYNC_MOVE_XY':
            len_half = 0.5 * self.ego_vehicles[ego_id].length
            position_sumo = [0, 0]
            position_sumo[0] = planned_state.position[0] + len_half * math.cos(planned_state.orientation)
            position_sumo[1] = planned_state.position[1] + len_half * math.sin(planned_state.orientation)
            sumo_angle = 90 - math.degrees(planned_state.orientation)

            self.vehicledomain.moveToXY(id_sumo,
                                        edgeID='dummy',
                                        laneIndex=-1,
                                        x=position_sumo[0],
                                        y=position_sumo[1],
                                        angle=sumo_angle,
                                        keepRoute=2)

            self.vehicledomain.setSpeedMode(id_sumo, 0)
            self.vehicledomain.setSpeed(id_sumo, planned_state.velocity)

        elif sync_mechanism == 'SYNC_SUMO_R_LC':
            # A lane change (right lane change) is just started, so we will initiate lane change request by traci
            # self.vehicledomain.setLaneChangeDuration(cr2sumo(default_ego_id, self.ids_cr2sumo), lc_duration)
            self.vehicledomain.setLaneChangeMode(id_sumo, 512)
            targetlane = self.vehicledomain.getLaneIndex(id_sumo) - 1
            self.vehicledomain.changeLane(id_sumo, targetlane, 0.1)
        elif sync_mechanism == 'SYNC_SUMO_L_LC':
            # A lane change (left lane change) is just started, so we will initiate lane change request by traci
            # self.vehicledomain.setLaneChangeDuration(cr2sumo(default_ego_id, self.ids_cr2sumo), lc_duration)
            self.vehicledomain.setLaneChangeMode(id_sumo, 512)
            targetlane = self.vehicledomain.getLaneIndex(id_sumo) + 1
            self.vehicledomain.changeLane(id_sumo, targetlane, 0.1)
        elif sync_mechanism == 'SYNC_DO_NOTHING':
            pass
        else:
            pass

    def is_in_field_of_view(self, position: np.ndarray, veh_id: str):
        """
        Returns True if the position is in the FOV of any ego vehicle.
        :param position: position of other vehicle
        :return:
        """
        if len(self.ego_vehicles) == 0 or self.conf.field_of_view is None \
            or veh_id in self.ids_sumo2cr[SUMO_VEHICLE_PREFIX] or veh_id in self.ids_sumo2cr[SUMO_PEDESTRIAN_PREFIX]:
            return True

        for ego_vehicle in self.ego_vehicles.values():
            try:
                if np.linalg.norm(ego_vehicle.get_state_at_timestep(self.current_time_step).position - position,
                                  ord=np.inf) < self.conf.field_of_view:
                    return True
            except KeyError:
                if len(ego_vehicle._state_dict.keys()) == 0:
                    raise ValueError

        return False

    @staticmethod
    def sort_ego_first(vehicle_ids: List[str]):
        vehicle_ids = list(vehicle_ids)
        for i in range(len(vehicle_ids)):
            if vehicle_ids[i].startswith(EGO_ID_START):
                vehicle_ids.insert(0, vehicle_ids.pop(i))
        return vehicle_ids

