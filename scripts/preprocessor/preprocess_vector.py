import abc

from pyquaternion import Quaternion
import numpy as np
from typing import Dict, Tuple, Union, List
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os

from preprocessor.preprocessor import PreprocessInterface
from preprocessor.utils import AgentType, TrafficLight
from utils.angle_operation import get_normalized_angle
from utils.transform import XYYawTransform

class PreprocessVectorization(PreprocessInterface):
    """
    Class for agent prediction, using the vector representation for maps and agents
    """
    def __init__(self, args: Dict):
        super().__init__()

        self.t_h: float= args['t_history']
        self.t_f: float= args['t_future']

        args = args['vectorize']
        self.t_interval: float= args['t_interval']
        self.feat_siz: int= args['feature_size']

        self.map_extent = args['map_extent']
        self.polyline_resolution: float= args['polyline_resolution']
        self.polyline_length: int= args['polyline_length']

        self.max_nodes: int= args['num_lane_nodes']
        self.max_vehicles_num: int= args['num_vehicles']
        self.max_pedestrians_num: int= args['num_pedestrians']

        self.t_h_state_num: int= int(self.t_h / self.t_interval) + 1 # add one current state
        self.t_f_state_num: int= int(self.t_f / self.t_interval)

    ### Overrides
    def extract_target_agent_representation(self, idx: int, agent_idx: int) -> np.ndarray:
        """
        Extracts target agent representation
        :param idx: data index
        :param agent_idx: agent index
        :return hist: track history for target agent, shape: [t_h * 2 + 1, feat_siz]
        """
        ori_pose = self.get_target_agent_global_pose(idx, agent_idx)

        # x, y co-ordinates in agent's frame of reference
        hist = self.get_past_motion_states(idx, ori_pose, agent_idx, in_agent_frame=True)

        # Zero pad for track histories shorter than t_h
        hist_zeropadded = np.zeros((self.t_h_state_num, self.feat_siz))
        hist_zeropadded[-hist.shape[0]:] = hist

        hist = hist_zeropadded
        return hist

    def extract_map_representation(self, idx: int, agent_idx: int) -> Union[int, Dict]:
        """
        Extracts map representation
        :param idx: data index
        :param agent_idx: agent index
        :return: Returns an ndarray with lane node features, shape 
                 [max_nodes, polyline_length, feat_siz] and an ndarray of
                 masks of the same shape, with value 1 if the nodes/poses are empty,
        """
        # Get agent representation in global co-ordinates
        global_pose = self.get_target_agent_global_pose(idx, agent_idx)

        # Get lanes around agent within map_extent
        lanes = self.get_lanes_around_agent(global_pose, self.polyline_resolution)

        # Get relevant polygon layers from the map_api
        polygons = self.get_polygons_around_agent(global_pose)
        traffic_lights = self.get_traffic_lights_around_agent(global_pose)

        # Get vectorized representation of lanes
        lane_node_feats, _ = self.get_lane_node_feats(global_pose, lanes, polygons, traffic_lights)

        # Discard lanes outside map extent
        lane_node_feats = self.discard_poses_outside_extent(lane_node_feats)

        # Add dummy node (0, 0, 0, 0, 0) if no lane nodes are found
        if len(lane_node_feats) == 0:
            lane_node_feats = [np.zeros((1, self.feat_siz))]

        # Convert list of lane node feats to fixed size numpy array and masks
        lane_node_feats, lane_node_masks =\
            self.list_to_tensor(lane_node_feats, self.max_nodes, self.polyline_length, self.feat_siz)

        map_representation = {
            'lane_node_feats': lane_node_feats,
            'lane_node_masks': lane_node_masks,
        }

        return map_representation

    def extract_surrounding_agent_representation(self, idx: int, agent_idx: int) -> \
            Union[Tuple[int, int], Dict]:
        """
        Extracts surrounding agent representation
        :param idx: data index
        :param agent_idx: agent index
        :return: ndarrays with surrounding pedestrian and vehicle track histories and masks for non-existent agents
        """

        # Get vehicles and pedestrian histories for current sample
        vehicles = self.get_surrounding_agents_of_type(idx, agent_idx, AgentType.VEHICLE)
        pedestrians = self.get_surrounding_agents_of_type(idx, agent_idx, AgentType.PEDESTRIAN)
        # Discard poses outside map extent
        vehicles = self.discard_poses_outside_extent(vehicles)
        pedestrians = self.discard_poses_outside_extent(pedestrians)

        # Convert to fixed size arrays for batching
        vehicles, vehicle_masks = self.list_to_tensor(
            vehicles, self.max_vehicles_num, self.t_h_state_num, self.feat_siz)
        pedestrians, pedestrian_masks = self.list_to_tensor(
            pedestrians, self.max_pedestrians_num, self.t_h_state_num, self.feat_siz)

        surrounding_agent_representation = {
            'vehicles': vehicles,
            'vehicle_masks': vehicle_masks,
            'pedestrians': pedestrians,
            'pedestrian_masks': pedestrian_masks
        }

        return surrounding_agent_representation

    def extract_target_representation(self, idx: int, agent_idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts target representation for target agent
        :param idx: data index
        :param agent_idx: agent index
        """
        return self.get_future_motion_states(idx, agent_idx, 
            involve_full_future=False, in_agent_frame=True)

    ### Abstracts
    @abc.abstractmethod
    def get_past_motion_states(self, idx: int, ori_pose: Tuple, 
                                     agent_idx: int, in_agent_frame: bool) -> np.ndarray:
        '''
        Extract target agent past history
        :param idx: data index
        :param ori_pose: original agent pose
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
    def get_target_agent_global_pose(self, idx: int, agent_idx: int) -> Tuple[float, float, float]:
        """
        Returns global pose of target agent
        :param idx: data index
        :param agent_idx: agent index
        :return global_pose: (x, y, yaw) or target agent in global co-ordinates
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_lanes_around_agent(self, global_pose: Tuple[float, float, float], 
                                     polyline_resolution: float) -> Dict:
        """
        Gets lane polylines around the target agent
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :param polyline_resolution: resolution to sample lane centerline poses
        :return lanes: Dictionary of lane polylines (centerline)
            lane_key(str or int) > List[Tuple(float, float, float)]
                                    [(x, y, yaw)_{lane_node0}, (x, y, yaw)_{1}, ...]
        @note yaw value is critical when graph processing.!
        """
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
    def get_traffic_lights_around_agent(self, global_pose: Tuple[float, float, float]) -> Dict[TrafficLight, List[Polygon]]:
        """
        Gets traffic light layers around the target agent
        :param global_pose: (x, y, yaw) or target agent in global co-ordinates
        :return dict of traffic lights:
            {   
                TrafficLight: List[Polygon],
            }
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_surrounding_agent_indexs_with_type(self, idx: int, agent_idx: int, agent_type: AgentType) -> List[int]:
        """
        Returns surrounding agents's list of indexs
        :param idx: data index
        :param agent_idx: agent index
        :param agent_type: AgentType
        :return: list of indexs of surrounding agents
        """
        raise NotImplementedError()

    ### Class functions
    def get_lane_node_feats(self, origin: Tuple, lanes: Dict[Union[str, int], List[Tuple]],
                            polygons: Dict[str, List[Polygon]],
                            traffic_lights: Dict[TrafficLight, List[Polygon]]
        ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Generates vector HD map representation in the agent centric frame of reference
        :param origin: (x, y, yaw) of target agent in global co-ordinates
        :param lanes: lane centerline poses in global co-ordinates
        :param polygons: stop-line and cross-walk polygons in global co-ordinates
        :return:
        """

        # Convert lanes to list
        lane_ids = [k for k, v in lanes.items()]
        lanes = [v for k, v in lanes.items()]

        # Get flags indicating whether a lane lies on 
        # 1. stop lines or crosswalks:
        # 2. traffic lights
        # return List[ np.array=(lane_node_num, 2=flags) ]
        lane_flags = self.get_lane_flags(lanes, polygons, traffic_lights)

        # Convert lane polylines to local coordinates: [num, 6, 3]
        lanes = [np.asarray([self.global_to_local(origin, pose) for pose in lane]) for lane in lanes]

        # Concatenate lane poses and lane flags
        lane_node_feats = [np.concatenate((lanes[i], lane_flags[i]), axis=1) for i in range(len(lanes))]

        # Split lane centerlines into smaller segments:
        lane_node_feats, lane_node_ids = self.split_lanes(lane_node_feats, self.polyline_length, lane_ids)

        return lane_node_feats, lane_node_ids

    def get_surrounding_agents_of_type(self, idx: int, agent_idx: int, agent_type: AgentType) -> List[np.ndarray]:
        """
        Returns surrounding agents of a particular class for a given sample
        :param idx: data index
        :param agent_idx: agent index
        :param agent_type: AgentType
        :return: list of ndarrays of agent track histories.
             List[ agents (in assigned type) with data from get_past_motion_states() ]
        """
        agent_list = []

        ori_pose = self.get_target_agent_global_pose(idx, agent_idx)

        indexs = self.get_surrounding_agent_indexs_with_type(idx, agent_idx, agent_type)
        for _agent_idx in indexs:
            agent_list.append(
                self.get_past_motion_states(
                    idx, ori_pose, _agent_idx, in_agent_frame=True))

        return agent_list

    def discard_poses_outside_extent(self, pose_set: List[np.ndarray],
                                     ids: List[str] = None) -> Union[List[np.ndarray],
                                                                     Tuple[List[np.ndarray], List[str]]]:
        """
        Discards lane or agent poses outside predefined extent in target agent's frame of reference.
        :param pose_set: agent or lane polyline poses
        :param ids: annotation record tokens for pose_set. Only applies to lanes.
        :return: Updated pose set
        """
        updated_pose_set = []
        updated_ids = []

        for m, poses in enumerate(pose_set):
            flag = False
            for n, pose in enumerate(poses):
                if self.map_extent[0] <= pose[0] <= self.map_extent[1] and \
                        self.map_extent[2] <= pose[1] <= self.map_extent[3]:
                    flag = True

            if flag:
                updated_pose_set.append(poses)
                if ids is not None:
                    updated_ids.append(ids[m])

        updated_pose_set = updated_pose_set[:self.max_nodes]
        updated_ids = updated_ids[:self.max_nodes]

        if ids is not None:
            return updated_pose_set, updated_ids
        else:
            return updated_pose_set

    @staticmethod
    def global_to_local(origin: Tuple, global_pose: Tuple) -> Tuple:
        """
        Converts pose in global co-ordinates to local co-ordinates.
        :param origin: (x, y, yaw) of origin in global co-ordinates
        :param global_pose: (x, y, yaw) in global co-ordinates
        :return local_pose: (x, y, yaw) in local co-ordinates
        """
        # Unpack
        global_x, global_y, global_yaw = global_pose
        origin_x, origin_y, origin_yaw = origin

        # Translate + Rotate
        inv_origin = XYYawTransform(x=origin_x, y=origin_y, yaw_radian=origin_yaw)
        inv_origin.inverse()
        get_pose = inv_origin.multiply_from_right(
            XYYawTransform(x=global_x, y=global_y, yaw_radian=global_yaw)
        )
        local_pose = (get_pose._x, get_pose._y, get_pose._yaw)

        return local_pose

    @staticmethod
    def split_lanes(lanes: List[np.ndarray], max_len: int, 
                    lane_ids: List[Union[str, int]]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Splits lanes into roughly equal sized smaller segments with defined maximum length
        :param lanes: list of lane poses
        :param max_len: maximum admissible length of polyline
        :param lane_ids: list of lane ID tokens
        :return lane_segments: list of smaller lane segments
                lane_segment_ids: list of lane ID tokens corresponding to original lane that the segment is part of
        """
        lane_segments = []
        lane_segment_ids = []
        for idx, lane in enumerate(lanes):
            n_segments = int(np.ceil(len(lane) / max_len))
            n_poses = int(np.ceil(len(lane) / n_segments))
            for n in range(n_segments):
                lane_segment = lane[n * n_poses: (n+1) * n_poses]
                lane_segments.append(lane_segment)
                lane_segment_ids.append(lane_ids[idx])

        return lane_segments, lane_segment_ids

    def get_lane_flags(self, lanes: List[List[Tuple]], 
                             polygons: Dict[str, List[Polygon]],
                             traffic_lights: Dict[TrafficLight, List[Polygon]]
        ) -> List[np.ndarray]:
        """
        Returns flags indicating whether each pose on lane polylines lies on polygon map layers
        like stop-lines or cross-walks
        :param lanes: list of lane poses
        :param polygons: dictionary of polygon layers
        :return lane_flags: list of ndarrays with flags
        """
        poly_value = self.poly_priority_idx
        tl_value = self.tl_color_to_priority_idx

        lane_flags = [np.zeros((len(lane), 2)) for lane in lanes]
        for lane_num, lane in enumerate(lanes):
            for pose_num, pose in enumerate(lane):
                point = Point(pose[0], pose[1])
                # for n, k in enumerate(polygons.keys()):
                #     # two type of polygon, 
                #     polygon_list = polygons[k]
                #     for polygon in polygon_list:
                #         if polygon.contains(point):
                #             lane_flags[lane_num][pose_num][n] = 1
                #             break
                for poly_key, polygon_list in polygons.items():
                    for polygon in polygon_list:
                        if polygon.contains(point):
                            lane_flags[lane_num][pose_num][0] += poly_value[poly_key]
                            break # one poly_key add lane_node value once

                for light_key, polygon_list in traffic_lights.items():
                    enable_brake = False
                    for polygon in polygon_list:
                        if polygon.contains(point):
                            lane_flags[lane_num][pose_num][1] = tl_value[light_key]
                            enable_brake = True
                            break
                    if enable_brake:
                        break # one traffic light set value of lane_node once

        return lane_flags

    @staticmethod
    def list_to_tensor(feat_list: List[np.ndarray], max_num: int, max_len: int,
                       feat_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a list of sequential features (e.g. lane polylines or agent history) to fixed size numpy arrays for
        forming mini-batches

        :param feat_list: List of sequential features
        :param max_num: Maximum number of sequences in List
        :param max_len: Maximum length of each sequence
        :param feat_size: Feature dimension
        :return: 1) ndarray of features of shape [max_num, max_len, feat_dim]. Has zeros where elements are missing,
            2) ndarray of binary masks of shape [max_num, max_len, feat_dim]. Has ones where elements are missing.
        """
        feat_array = np.zeros((max_num, max_len, feat_size))
        mask_array = np.ones((max_num, max_len, feat_size))
        for n, feats in enumerate(feat_list):
            feat_array[n, :len(feats), :] = feats
            mask_array[n, :len(feats), :] = 0

        return feat_array, mask_array

    @staticmethod
    def flip_horizontal(data: Dict):
        """
        Helper function to randomly flip some samples across y-axis for data augmentation
        :param data: Dictionary with inputs and ground truth values.
        :return: data: Dictionary with inputs and ground truth values fligpped along y-axis.
        """
        # Flip target agent
        hist = data['inputs']['target_agent_representation']
        hist[:, 0] = -hist[:, 0]  # x-coord
        hist[:, 4] = -hist[:, 4]  # yaw-rate
        data['inputs']['target_agent_representation'] = hist

        # Flip lane node features
        lf = data['inputs']['map_representation']['lane_node_feats']
        lf[:, :, 0] = -lf[:, :, 0]  # x-coord
        lf[:, :, 2] = -lf[:, :, 2]  # yaw
        data['inputs']['map_representation']['lane_node_feats'] = lf

        # Flip surrounding agents
        vehicles = data['inputs']['surrounding_agent_representation']['vehicles']
        vehicles[:, :, 0] = -vehicles[:, :, 0]  # x-coord
        vehicles[:, :, 4] = -vehicles[:, :, 4]  # yaw-rate
        data['inputs']['surrounding_agent_representation']['vehicles'] = vehicles

        peds = data['inputs']['surrounding_agent_representation']['pedestrians']
        peds[:, :, 0] = -peds[:, :, 0]  # x-coord
        peds[:, :, 4] = -peds[:, :, 4]  # yaw-rate
        data['inputs']['surrounding_agent_representation']['pedestrians'] = peds

        # Flip groud truth trajectory
        fut = data['ground_truth']['traj']
        fut[:, 0] = -fut[:, 0]  # x-coord
        data['ground_truth']['traj'] = fut

        return data
