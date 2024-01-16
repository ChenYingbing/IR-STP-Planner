from pyquaternion import Quaternion
import numpy as np
from typing import Dict, Tuple, Union, List
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import math

from preprocessor.preprocess_vector import PreprocessVectorization
from preprocessor.utils import AgentType, TrafficLight

import thirdparty.config
from commonroad.scenario.scenario import Scenario
from utils.colored_print import ColorPrinter
from utils.transform import XYYawTransform
from utils.angle_operation import get_normalized_angle
from commonroad.scenario.obstacle import ObstacleType

class CommonroadVectorizer(PreprocessVectorization):
    """
    Class for agent prediction, using the vector representation for maps and agents
    """
    def __init__(self, args: Dict):
        super().__init__(args=args)
        self.map2agent_type = {
            ObstacleType.CAR: AgentType.VEHICLE,
            ObstacleType.BUS: AgentType.VEHICLE,
            ObstacleType.TRUCK: AgentType.VEHICLE,
            ObstacleType.BICYCLE: AgentType.VEHICLE,
            ObstacleType.MOTORCYCLE: AgentType.VEHICLE,
            ObstacleType.PEDESTRIAN: AgentType.PEDESTRIAN,
        }

        self.search_radius = max([math.fabs(v) for v in self.map_extent])

        self.feat_dim2func :Dict = {
            5 : self.get_extra_feature_5_dim,
        }

        self.enable_rviz = False

    def init_scenario(self, scenario: Scenario, enable_rviz: bool=False):
        '''
        Set current scenario and frame index
        '''
        self.scenario: Scenario = scenario
        self.state_interval :int= int(self.t_interval / scenario.dt)

        self.traffic_lights = scenario.lanelet_network.traffic_lights
        # definition: speed limit, stop, yields, no-parking ..
        self.traffic_signs = scenario.lanelet_network.traffic_signs
        self.intersections = scenario.lanelet_network.intersections
        self.obstacle_list = {obstacle.obstacle_id: obstacle for obstacle in scenario.obstacles}

        self.enable_rviz = enable_rviz
        # cache values for operation
        self.cache_lanelet_list = None
        self.cache_lanelet_dict = None

        # for lanelet in scenario.lanelet_network.lanelets:
        #   if lanelet.stop_line == True:
        #   if len(lanelet.traffic_lights) > 0:
        #   print("ID", lanelet.lanelet_id)
        #   print("stop_line", lanelet.stop_line)
        #   print("traffic_signs", lanelet.traffic_signs)
        #   print("traffic_lights", lanelet.traffic_lights)
        #   # TrafficSignIDBelgium.MAX_SPEED ['13.88888888888889']
        #   # TrafficSignIDGermany.SIDEWALK []
        #   # TrafficSignIDGermany.YIELD []
        #   # TrafficSignIDGermany.BUS_STOP []

    def init_frame_from_to(self, frame_from: int, frame_to: int):
        # print("process frame from={}, to={}.".format(frame_from, frame_to))
        self.frame_from: int = frame_from
        self.frame_to: int = frame_to

    def get_agent_valid_state_range(self, obstacle):
        '''
        Obtain obstacle relative state index in the self.frame_from
        :param obstacle: type of obstacle in commonroad
        return bool, [state_from, state_to]
            bool: when it is true, the obstacle has valid state in [frame_from, frame_to]
            state_from/to:  used in 
                @note state_list[0] == agent.initial_state
        '''
        state_list = obstacle.prediction.trajectory.state_list
        frame0: int= obstacle.initial_state.time_step
        frame1: int= frame0 + len(state_list)

        if  (self.frame_from < frame0) or (self.frame_from >= frame1):
            return False, [frame0, frame1] # time not overlapped
        else:
            state_from = self.frame_from - frame0
            state_to = min(self.frame_to - frame0, frame1)
            return True, [state_from, state_to]

    @staticmethod
    def get_extra_feature_5_dim(input_state):
        # print("input", input_state.velocity)
        return [input_state.velocity, 0.0]

    ### Abstracts
    def get_past_motion_states(self, idx: int, ori_pose: Tuple, 
                                     agent_idx: int, in_agent_frame: bool) -> np.ndarray:
        '''
        Extract target agent past history
        :param idx: data index
        :param ori_pose: original pose
        :param agent_idx: agent index
        :param in_agent_frame: representation in agent frame
        return shape = (valid_history + 1 , feat_siz)
        [
          ...,
          [x, y, yaw, v, yaw_rate, ...]_{t = current - t_interval * 2.0},
          [x, y, yaw, v, yaw_rate, ...]_{t = current - t_interval * 1.0},
          [x, y, yaw, v, yaw_rate, ...]_{t = current},
        ]
        '''
        extra_feat_func = self.feat_dim2func[self.feat_siz]

        agent = self.obstacle_list[agent_idx]
        flag, state_range = self.get_agent_valid_state_range(agent)
        assert flag == True, "Fatal error1, fails to get state indexs"
        state_idx_now = state_range[0]
        state_seq = list(range(state_idx_now, -1, -self.state_interval)
            )[:self.t_h_state_num] # assert state_idx_now is added
        state_seq.reverse() # in reverse seq

        # state_now = state_list[state_idx_now]
        state_list = agent.prediction.trajectory.state_list

        past_states = []
        if in_agent_frame == False:
            for k in state_seq:
                state_k = state_list[k]
                past_states.append(
                    [state_k.position[0], state_k.position[1], state_k.orientation] +\
                        extra_feat_func(state_k)) # fill x, y, yaw, v
        else:
            for k in state_seq:
                state_k = state_list[k]
                past_states.append(
                    list(self.global_to_local(ori_pose,
                            (state_k.position[0], state_k.position[1], state_k.orientation))
                    ) + extra_feat_func(state_k)) # fill x, y, yaw, v
        past_states = np.array(past_states)
        
        # print("past_states", state_idx_now, state_seq, past_states.shape)
        if self.feat_siz == 5:
            dyaws = past_states[1:, 2] - past_states[:-1, 2]
            dyaw_rates = np.array([(get_normalized_angle(dyaw)/self.t_interval) for dyaw in dyaws])
            past_states[:-1, 4] = dyaw_rates # fill yaw rates
        else:
            raise NotImplementedError("Not implemented")
        
        return past_states

    def get_future_motion_states(self, idx: int, agent_idx: int, in_agent_frame: bool, 
                                       involve_full_future: bool=False) -> np.ndarray:
        '''
        Extract target agent future history
        :param idx: data index
        :param agent_idx: agent index
        :param in_agent_frame: representation in agent frame
        :param involve_full_future: contain all future states (not just for prediction)
        return shape = (valid_future , feat_siz)
        [
          [x, y, yaw, v, yaw_rate, ...]_{t = current + t_interval * 1.0},
          [x, y, yaw, v, yaw_rate, ...]_{t = current + t_interval * 2.0},
        ]
        '''
        extra_feat_func = self.feat_dim2func[self.feat_siz]

        agent = self.obstacle_list[agent_idx]
        flag, state_range = self.get_agent_valid_state_range(agent)
        assert flag == True, "Fatal error2, fails to get state indexs"
        
        state_idx_now = state_range[0]
        state_idx_to = state_range[1]
        state_list = agent.prediction.trajectory.state_list
        assert len(state_list) >= state_idx_to, "Fatal error3, fails to get state indexs"

        state_seq = []
        if involve_full_future == False:
            state_seq = list(range(state_idx_now + self.state_interval, 
                                state_idx_to+1, self.state_interval))
            assert len(state_seq) == self.t_f_state_num, "Fatal error4, fails to get state indexs"
        else:
            state_seq = list(range(state_idx_now + self.state_interval, 
                                len(state_list), self.state_interval))

        state_now = state_list[state_idx_now]
        agent_pose = (state_now.position[0], state_now.position[1], state_now.orientation)
        future_states = []
        if in_agent_frame == False:
            for k in state_seq:
                state_k = state_list[k]
                future_states.append(
                    [state_k.position[0], state_k.position[1], state_k.orientation] +\
                        extra_feat_func(state_k)) # fill x, y, yaw, v
        else:
            for k in state_seq:
                state_k = state_list[k]
                future_states.append(
                    list(self.global_to_local(agent_pose,
                            (state_k.position[0], state_k.position[1], state_k.orientation))
                    ) + extra_feat_func(state_k)) # fill x, y, yaw, v
        future_states = np.array(future_states)

        return future_states

    def get_target_agent_global_pose(self, idx: int, agent_idx: int) -> Tuple[float, float, float]:
        """
        Returns global pose of target agent
        :param idx: data index
        :param agent_idx: agent index
        :return global_pose: (x, y, yaw) or target agent in global co-ordinates
        """
        agent = self.obstacle_list[agent_idx]

        flag, state_range = self.get_agent_valid_state_range(agent)
        assert flag == True, "get_target_agent_global_pose: Fatal error."
        
        state_now = agent.prediction.trajectory.state_list[state_range[0]]
        agent_pose = (state_now.position[0], state_now.position[1], state_now.orientation)

        # Cache data for later operation
        # self.cache_lanelet_list = self.scenario.lanelet_network.lanelets
        self.cache_lanelet_list = \
            self.scenario.lanelet_network.lanelets_in_proximity(
                np.array([state_now.position[0], state_now.position[1]]), 
                radius=self.search_radius)
        self.cache_lanelet_dict = {
            lanelet.lanelet_id : lanelet for lanelet in self.cache_lanelet_list
        }

        return agent_pose

    def get_lanes_around_agent(self, global_pose: Tuple[float, float, float], 
                                     polyline_resolution: float) -> Dict:
        """
        Gets lane polylines around the target agent
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param polyline_resolution: resolution to sample lane centerline poses
        :return lanes: Dictionary of lane polylines (centerline)
            lane_key(str or int) > List[Tuple(float, float, float)]
                                    [(x, y, yaw)_{lane_node0}, (x, y, yaw)_{1}, ...]
        """
        dict_lanes: Dict[int, List] = {}

        for _, lanelet in enumerate(self.cache_lanelet_list):
            center_points = lanelet.center_vertices # [[x, y], ...]
            lane_s_array = lanelet.distance
            lane_s_from = lane_s_array[0]
            lane_s_to = lane_s_array[-1]
            lane_s_range = lane_s_to - lane_s_from

            piece_num = math.ceil(lane_s_range / polyline_resolution)
            lane_points = [(tuple(
                lanelet.interpolate_position(
                    min(lane_s_from + lane_s_range * float(pi) / piece_num, lane_s_to)
                )[0].tolist()) + (0, ))
                 for pi in range(0, (piece_num+1), 1)]
            
            # calculate lane node yaws
            cache_xys = np.asarray(lane_points)
            lane_point_yaws = np.arctan2(cache_xys[1:, 1] - cache_xys[:-1, 1], 
                                         cache_xys[1:, 0] - cache_xys[:-1, 0])
            lane_point_yaws = lane_point_yaws.tolist() + [lane_point_yaws[-1]]
            lane_xyyaws = [(point[0], point[1], yaw) for point, yaw in zip(lane_points, lane_point_yaws)]

            dict_lanes[lanelet.lanelet_id] = lane_xyyaws

        # print("dict_lanes", dict_lanes.keys())
        return dict_lanes

    def get_polygons_around_agent(self, global_pose: Tuple[float, float, float]) -> Dict:
        """
        Gets polygon layers around the target agent e.g. crosswalks, stop lines
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :return polygons: Dictionary of polygon layers, each type as a list of shapely Polygons
            {   
                'stop_line': List[Polygon],
                'ped_crossing': List[Polygon]
            }
        """
        # if len(self.intersections) > 0:
        # @note intersection relevant lanes are also available in self.cache_lanelet_list
        stop_line_reso = self.polyline_resolution * 0.5

        sl_polygons = []
        for lanelet in self.cache_lanelet_list:
            if lanelet.stop_line:
                start_xy = lanelet.stop_line.start
                end_xy = lanelet.stop_line.end
                direct = math.atan2(end_xy[1]-start_xy[1], end_xy[0]-start_xy[0])
                
                mid_xy = [
                    0.5 * (start_xy[0] + end_xy[0]),
                    0.5 * (start_xy[1] + end_xy[1]),
                ]
                mid_xy = XYYawTransform(x=mid_xy[0], y=mid_xy[1], yaw_radian=direct)
                left_xy = mid_xy.multiply_from_right(XYYawTransform(y=stop_line_reso))
                right_xy = mid_xy.multiply_from_right(XYYawTransform(y=-stop_line_reso))
                sl_polygons.append(Polygon(tuple([
                    start_xy, [right_xy._x, right_xy._y], end_xy, [left_xy._x, left_xy._y]])))

        return {
            'stop_line': sl_polygons,
            'ped_crossing': [],
        }

    def get_traffic_lights_around_agent(self, global_pose: Tuple[float, float, float]) -> Dict[TrafficLight, List[Polygon]]:
        """
        Gets traffic light layers around the target agent
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :return dict of traffic lights:
            {   
                TrafficLight: List[Polygon],
            }
        """
        # for lanelet in self.cache_lanelet_list:
        #     if len(lanelet.traffic_lights) > 0:
        #         pass 
        # TODO: traffic light is not envolved in commonroad
        return {}

    def get_surrounding_agent_indexs_with_type(self, idx: int, agent_idx: int, agent_type: AgentType) -> List[int]:
        """
        Returns surrounding agents's list of indexs, agent should be type of agent_type
        :param idx: data index
        :param agent_idx: agent index
        :param agent_type: AgentType
        :return: list of indexs of surrounding agents
        """
        target_agent = self.obstacle_list[agent_idx]
        target_agent_state = target_agent.initial_state
        inv_xyyaw = XYYawTransform(x=target_agent_state.position[0],
                                   y=target_agent_state.position[1],
                                   yaw_radian=target_agent_state.orientation)
        inv_xyyaw.inverse()

        agent_list = []
        for _idx, agent in self.obstacle_list.items():
            if _idx == agent_idx:
                continue # skip when is target agent

            if self.map2agent_type[agent.obstacle_type] != agent_type:
                continue # skip when is not required type

            flag, state_range = self.get_agent_valid_state_range(agent)
            if flag == False:
                continue # skip when agent is invalid

            state_from = state_range[0]
            state_to = state_range[1]
            agent_state = agent.prediction.trajectory.state_list[state_from]
            
            dpose = inv_xyyaw.multiply_from_right(
                XYYawTransform(x=agent_state.position[0],
                               y=agent_state.position[1],
                               yaw_radian=agent_state.orientation))
            distance = math.sqrt(dpose._x**2 + dpose._y**2)
            if distance > self.search_radius:
                continue # skip when is out of search distance

            agent_list.append(_idx)
            # print(self.frame_from, self.frame_to)

        # print("get_surrounding_agent_indexs_with_type: ", agent_list)
        return agent_list
