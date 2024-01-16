from ast import Break

from enum import IntEnum
from utils.colored_print import ColorPrinter
from utils.angle_operation import get_normalized_angle
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
from scipy.spatial import KDTree
from typing import Dict, List, Any, Tuple, Union
import numpy as np
import math
import networkx
import matplotlib.pyplot as plt
import copy

import thirdparty.config
from commonroad.scenario.lanelet import LaneletNetwork
from preprocessor.utils import TrafficLight
from utils.transform import XYYawTransform
from type_utils.frenet_path import FrenetPath

TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE

class RouteEdgeType(IntEnum):
  FOLLOW = 0,
  LEFT_CHANGE = 1,
  RIGHT_CHANGE = 2,

class NavigationMap:
  def __init__(self, polyline_reso:float=1.0,
                     graph_succ_edge_cost:float=0.5,
                     graph_left_edge_cost:float=1.0,
                     graph_right_edge_cost:float=1.0) -> None:
    '''
    Init parameters for navigation map
    :param polyline_reso: polyline resolution (meter)
    '''
    self.clear()

    self.polyline_reso = polyline_reso
    self.graph_succ_edge_cost = graph_succ_edge_cost
    self.graph_left_edge_cost = graph_left_edge_cost
    self.graph_right_edge_cost = graph_right_edge_cost

  def clear(self):
    # graph
    self.lane_graph = networkx.DiGraph()
    self.lane_graph_node_xys: Dict[int, np.array] = {} # for visualization
    self.lane_mid_points: Dict[int, np.array] = {} # lane index: lane node array
    self.lane_left_points: Dict[int, np.array] = {} # lane index: lane node array
    self.lane_right_points: Dict[int, np.array] = {} # lane index: lane node array
    self.lane_id_list: List = [] # list of lane idxs
    # @note lane_edges descibe relative positions of lanes regardless of 
    #       lane direction!
    # @example, lane 1 can be left to lane 2 but they are with opposite lane direction
    self.lane_edges: Dict[int, Dict[str, List]] = {} # lane index: {'pred': [], 'succ': [], 'left': [], 'right': []}

    # map: str_tag > List[semantic objs]
    self.scene_objects: Dict[str, List] = {}
    self.scene_objects['lane_id'] = set() # lane indexs
    self.scene_objects['tl'] = []   # traffic lights: List[{'color': TrafficLight, 'poly': Polygon}]
    self.scene_objects['cw'] = []   # crosswalks: List[Polygon]

    # map: str_tag > lane_points [x, y, yaw] (used in kdtree)
    self.kdt_lane_points: Dict[str, np.ndarray] = {}
    self.kdt_lane_points['mid'] = np.array([], dtype=np.float64).reshape(0, 3)
    self.kdt_lane_points['left'] = np.array([], dtype=np.float64).reshape(0, 3)
    self.kdt_lane_points['right'] = np.array([], dtype=np.float64).reshape(0, 3)
    self.kdt_lane_points['route'] = np.array([], dtype=np.float64).reshape(0, 3)

    # map: str_tag > lane_point_infos [num]
    self.kdt_lane_point_infos: Dict[str, List[Dict]] = {} # follows pinfo format in UpdateLaneInfos()
    self.kdt_lane_point_infos['mid'] = []
    self.kdt_lane_point_infos['route'] = []

    # map: str_tag > kdtree
    self.kdtrees: Dict[str, KDTree] = {}
    self.kdtrees['mid'] = None
    self.kdtrees['left'] = None
    self.kdtrees['right'] = None
    self.kdtrees['route'] = None

    # task
    self.start: Dict= {}
    self.goal: Dict= {}
    self.route_seq: List[Tuple[int, RouteEdgeType]] = []
    self.route_edges: List[Tuple[int, int]] = []

    self.inited = False

  def ready_for_plan(self) -> bool:
    if self.inited == False:
      ColorPrinter.print('red', 'Navmap is not ready for planning: inited == false.')
      return False
    if len(self.route_seq) == 0:
      ColorPrinter.print('red', 'Navmap is not ready for planning: route len == 0.')
      return False

    return True

  def InitWithLaneletNet(self, lanelet_net: LaneletNetwork,
                         verbose: bool = False) -> None:
    '''
    Init the map with lanelet network
    '''
    self.clear()
    self.lanelet_net = lanelet_net

    ### Initialize data
    list_lanelets = lanelet_net.lanelets

    # Tarverse semantic objs
    for lanelet in list_lanelets:
      self.scene_objects['lane_id'].add(lanelet.lanelet_id)
    self.scene_objects['tl'] = []
    self.scene_objects['cw'] = []

    # Traverse lanes
    for lanelet in list_lanelets:
      lane_s_array = lanelet.distance
      lane_s_from = lane_s_array[0]
      lane_s_to = lane_s_array[-1]
      lane_s_range = lane_s_to - lane_s_from

      piece_num = math.ceil(lane_s_range / self.polyline_reso)

      cps, rps, lps = [], [], []
      for pi in range(0, (piece_num+1), 1):
        cp, rp, lp, _ = lanelet.interpolate_position(
          min(lane_s_from + lane_s_range * float(pi) / piece_num, lane_s_to))
        cp = cp.tolist() + [0]
        rp = rp.tolist() + [0]
        lp = lp.tolist() + [0]
        cps.append(cp) # center points
        rps.append(rp) # right points
        lps.append(lp) # left points
      mid_points = self.update_yaw_values(np.asarray(cps)) # shape=(num_point, 3)
      right_points = self.update_yaw_values(np.asarray(rps)) # shape=(num_point, 3)
      left_points = self.update_yaw_values(np.asarray(lps)) # shape=(num_point, 3)

      # fill lane node xys
      self.lane_graph_node_xys[lanelet.lanelet_id] =\
        (mid_points[0, :2] + mid_points[-1, :2]) * 0.5
      self.lane_mid_points[lanelet.lanelet_id] = mid_points
      self.lane_left_points[lanelet.lanelet_id] = left_points
      self.lane_right_points[lanelet.lanelet_id] = right_points

      # fill lane points
      self.kdt_lane_points['mid'] = np.concatenate(
        (self.kdt_lane_points['mid'], mid_points), axis=0)
      self.kdt_lane_points['right'] = np.concatenate(
        (self.kdt_lane_points['right'], right_points), axis=0)
      self.kdt_lane_points['left'] = np.concatenate(
        (self.kdt_lane_points['left'], left_points), axis=0)

      # fill lane infos
      self.AppendLanePointInfo('mid', lanelet.lanelet_id, mid_points)

      # fill graph edge
      self.AppendGraphEdge(lanelet.lanelet_id, 'pred', lanelet.predecessor)
      self.AppendGraphEdge(lanelet.lanelet_id, 'succ', lanelet.successor)
      if lanelet.adj_left:
        self.AppendGraphEdge(lanelet.lanelet_id, 'left', [lanelet.adj_left])
      if lanelet.adj_right:
        self.AppendGraphEdge(lanelet.lanelet_id, 'right', [lanelet.adj_right])

    self.lane_id_list = self.lane_mid_points.keys()
    # connect graph edges
    for from_lane, neighbours in self.lane_edges.items():
      self.ConnectGraphEdge(from_lane, 'pred', neighbours['pred'])
      self.ConnectGraphEdge(from_lane, 'succ', neighbours['succ'])
      self.ConnectGraphEdge(from_lane, 'left', neighbours['left'])
      self.ConnectGraphEdge(from_lane, 'right', neighbours['right'])

    # fill kdtree
    self.kdtrees['mid'] = KDTree(self.kdt_lane_points['mid'][:, :2])
    self.kdtrees['left'] = KDTree(self.kdt_lane_points['left'][:, :2])
    self.kdtrees['right'] = KDTree(self.kdt_lane_points['right'][:, :2])

    if verbose:
      print("Init with kdt_lane_points['mid']=", self.kdt_lane_points['mid'].shape)
      print("Init with kdt_lane_points['right']=", self.kdt_lane_points['right'].shape)
      print("Init with kdt_lane_points['left']=", self.kdt_lane_points['left'].shape)
      print("Init with kdt_lane_point_infos['mid']=", len(self.kdt_lane_point_infos['mid']))
      print("Init graph node/edge num=", 
        self.lane_graph.number_of_nodes(), self.lane_graph.number_of_edges())

    self.inited = True

  def GetValidGoalOrientation(self, start: XYYawTransform, goal_x: float, goal_y: float) -> float:
    '''
    Get goal's orientation given goal's x and y position
    :param start: start point 
    :param goal_x: goal x position
    :param goal_y: goal y position
    '''
    # get start
    sinfo, _ = self.nearest_lane_and_point('mid', (start._x, start._y, start._yaw))
    start_lane_idx = sinfo['lane_idx']

    # check goals
    kdtree_key = 'mid'
    check_xy = [goal_x, goal_y]
    k_nearest_num = min(30, self.kdt_lane_points[kdtree_key].shape[0])
    nearest_dists, nearest_indexs =\
      self.kdtrees[kdtree_key].query((check_xy), k=k_nearest_num)
    if len(nearest_indexs) == 0:
      raise ValueError("Fails to check, the kd tree has none points")

    goal_orientation = None
    max_dist2d = 0.0
    for xy_dist, point_index in zip(nearest_dists, nearest_indexs):
      point = self.kdt_lane_points[kdtree_key][point_index, :]
      pinfo = self.kdt_lane_point_infos['mid'][point_index]

      has_path = self.check_lanes_has_path(start_lane_idx, pinfo['lane_idx'])
      if has_path:
        # dx = point[0] - start._x
        # dy = point[1] - start._y
        # dist = math.sqrt(dx**2 + dy**2)
        goal_orientation = point[2]
        break
    
    if goal_orientation == None:
      print("GetValidGoalOrientation::Error()")
      print("nearest_dists", nearest_dists)
      print("nearest_indexs", nearest_indexs)
      raise ValueError("Fails to get valid goal and its orientation")

    return goal_orientation

  def DepthFirstSearchGoal(self, start: XYYawTransform, search_depth: int=5) -> XYYawTransform:
    sinfo, sdetails = self.nearest_lane_and_point('mid', (start._x, start._y, start._yaw))

    start_lane_idx = sinfo['lane_idx']
    goal_lane_idx = copy.copy(start_lane_idx)
    while search_depth > 0:
      search_depth -= 1

      succ_list = self.get_lane_neighbours(goal_lane_idx)['succ']
      if len(succ_list) == 0:
        break
      goal_lane_idx = succ_list[0]

    goal_lane_points = self.lane_mid_points[goal_lane_idx]
    goal_xyyaw = goal_lane_points[-1, :] # assign last one as goal

    # ColorPrinter.print("yellow", "DepthFirstSearchGoal::Debug()")
    # print("sinfo", sinfo)
    # print("get to goal_lane_idx", start_lane_idx, goal_lane_idx, goal_lane_points.shape)
    # print("lanelet_id_keys", self.lane_id_list)
    
    return XYYawTransform(x=goal_xyyaw[0], y=goal_xyyaw[1], yaw_radian=goal_xyyaw[2])

  def InitRoute(self, start: XYYawTransform, goal: XYYawTransform, verbose: bool=False):
    '''
    Init route given start and goal position
    :return has_path
    '''
    # print("InitRoute::Debug::Start()")
    sinfo, sdetails = self.nearest_lane_and_point('mid', (start._x, start._y, start._yaw))
    # print("sinfo", sinfo, sdetails)
    # print("InitRoute::Debug::Goal()")
    ginfo, gdetails = self.nearest_lane_and_point('mid', (goal._x, goal._y, goal._yaw))
    # print("ginfo", ginfo, gdetails)
    # print("lane_id_list", self.lane_id_list)
    self.start['info'] = sinfo
    self.start['details'] = sdetails
    self.goal['info'] = ginfo
    self.goal['details'] = gdetails

    has_path, get_route = self.init_route(sinfo['lane_idx'], ginfo['lane_idx'])
    self.route_seq = get_route

    self.kdt_lane_points['route'] = np.array([], dtype=np.float64).reshape(0, 3)
    self.route_edges = []
    route_len = len(get_route)
    for ri in range(0, route_len):
      lane_idx: int= get_route[ri][0]
      if (ri + 1) < route_len:
        self.route_edges.append((lane_idx, get_route[ri+1][0]))
      self.kdt_lane_points['route'] = np.concatenate(
        (self.kdt_lane_points['route'], self.lane_mid_points[lane_idx]), axis=0)

      self.AppendLanePointInfo('route', lane_idx, self.lane_mid_points[lane_idx])
      
    # print("route points", self.kdt_lane_points['route'])
    self.kdtrees['route'] = KDTree(self.kdt_lane_points['route'][:, :2])

    if verbose:
      print("Init route sinfo", sinfo, sdetails)
      print("Init route ginfo", ginfo, gdetails)
      print("Init route with kdt_lane_points=", self.kdt_lane_points['route'].shape)
      print("Init route with kdt_lane_point_infos=", len(self.kdt_lane_point_infos['route']))
      print("Init_route has_path =", has_path)

    return has_path

  def AppendLanePointInfo(self, kdtree_key:str, lane_idx, lane_points: np.ndarray):
    '''
    Append lane point infos given lane_points
    :param kdtree_key: 'mid' or 'route'
    :param lane_points: shape = (lane_point_num, 3)
    '''
    lane_point_num = lane_points.shape[0]
    pinfo = {
      'lane_idx': lane_idx,               # lane index
      'kdt_idx': -1,                      # point index in kdtree
      'point_xyyaw': (None, None, None),  # point x, y, yaw value
      'lane_point_num': lane_point_num,   # lane's point amount
      'lane_point_idx': -1,               # point index at this lane
      'at_tl': [-1, TrafficLight.NONE],   # traffic light information [tl_index, tl_color]
      'at_cw': -1,                        # inside crosswalk index
    }

    index_from = len(self.kdt_lane_point_infos[kdtree_key])
    for local_idx in range(lane_point_num):
      point = lane_points[local_idx, :]
      pinfo['kdt_idx'] = (index_from + local_idx)
      pinfo['point_xyyaw'] = (point[0], point[1], point[2])
      pinfo['lane_point_idx'] = local_idx

      pp = Point(point[0], point[1])
      for tl_idx, tl in enumerate(self.scene_objects['tl']):
        tl_poly: Polygon = tl['poly']
        if tl_poly.contains(pp):
          pinfo['at_tl'] = [tl_idx, tl['color']]
          break
      
      for cw_idx, cw_poly in enumerate(self.scene_objects['cw']):
        if cw_poly.contains(pp):
          pinfo['at_cw'] = cw_idx
          Break

      # @note here must use copy operation
      self.kdt_lane_point_infos[kdtree_key].append(copy.copy(pinfo))

  def AppendGraphEdge(self, from_lane: int, edge_key: str, to_lane_list: List) -> None:
    '''
    Append lane edge records in self.lane_edges
    '''
    if not from_lane in self.lane_edges:
      self.lane_edges[from_lane] = {}
      self.lane_edges[from_lane]['pred'] = []
      self.lane_edges[from_lane]['succ'] = []
      self.lane_edges[from_lane]['left'] = []
      self.lane_edges[from_lane]['right'] = []

    for to_lane in to_lane_list:
      # Add edge record
      self.lane_edges[from_lane][edge_key].append(to_lane)

  def ConnectGraphEdge(self, from_lane: int, edge_key: str, to_lane_list: List) -> None:
    # print("ConnectGraphEdge::Debug()")
    # print(from_lane, edge_key, to_lane_list)
    for to_lane in to_lane_list:
      enable_connect = False
      edge_cost = 1e+3
      edge_color = 'k'
      if edge_key == 'pred':
        pass
      elif edge_key == 'succ':
        edge_cost = self.graph_succ_edge_cost
        edge_color = 'k'
        enable_connect = True
      else:
        edge_map = {
          'left': [self.graph_left_edge_cost, 'b'],
          'right': [self.graph_right_edge_cost, 'y']
        }
        edge_cost = edge_map[edge_key][0]
        edge_color = edge_map[edge_key][1]
        enable_connect = \
          self.check_left_right_lane_is_same_direction(from_lane, to_lane, edge_key)

      if enable_connect:
        self.lane_graph.add_edge(
          from_lane, to_lane, weight=edge_cost, color=edge_color)

  def nearest_lane_and_point(self, kdtree_key: str,
                                   check_point: Tuple[float, float, float], 
                                   meter_per_degree: float = TO_DEGREE*0.1,
                                   k_nearest_num: int = 30) -> Tuple:
    '''
    Calculate the nearest lane/point index
    :param kdtree_key: 'mid' or 'route'
    :param check_point: the input [x, y, yaw]
    :param meter_per_degree: coef to calcualte the distance of orientation
    :param k_nearest_num: number of point to query in kdtree
    :return point_info, [xy_dist, abs_dyaw]
    '''
    check_xy = check_point[0:2]
    check_yaw = check_point[2]

    # @note if k in query > maximum_number_of_points, the dist is inf, and index return is invalid.
    k_nearest_num = min(k_nearest_num, self.kdt_lane_points[kdtree_key].shape[0])
    # print("k_nearest_num=", check_xy, k_nearest_num, self.kdt_lane_points[kdtree_key].shape[0])

    nearest_dists, nearest_indexs =\
      self.kdtrees[kdtree_key].query((check_xy), k=k_nearest_num)
    if len(nearest_indexs) == 0:
      raise ValueError("Fails to check, the kd tree has none points")

    min_index = int(nearest_indexs[0])
    min_dist3d = np.inf
    min_xydist = None
    min_dyaw = None
    # print("nearest_lane_and_point::debug()")
    # print(kdtree_key, self.kdt_lane_points[kdtree_key].shape, len(self.kdt_lane_point_infos[kdtree_key]))
    # print("nearest_indexs", nearest_indexs)
    # print("nearest_dists", nearest_dists)
    for xy_dist, point_index in zip(nearest_dists, nearest_indexs):
      point = self.kdt_lane_points[kdtree_key][point_index, :]

      dyaw = get_normalized_angle(check_yaw - point[2])
      dist3d = xy_dist + math.fabs(dyaw) * meter_per_degree
      if dist3d < min_dist3d:
        min_index = int(point_index)

        min_dist3d = dist3d
        min_xydist = xy_dist
        min_dyaw = dyaw

    return self.kdt_lane_point_infos[kdtree_key][min_index], [min_xydist, min_dyaw]

  def check_lanes_has_path(self, from_lane: int, to_lane: int):
    '''
    Return true when there exists path between from_lane and to_lane in the lane_graph
    '''
    if not (self.lane_graph.has_node(from_lane) and self.lane_graph.has_node(to_lane)):
      return False
    if from_lane == to_lane:
      return True

    has_path = networkx.has_path(self.lane_graph, from_lane, to_lane)
    return has_path

  def init_route(self, from_lane: int, to_lane: int
    ) -> Tuple[bool, List[Tuple[int, RouteEdgeType]]]:
    '''
    init route given from/to lane index
    :param from_lane: is used calcuate the route only when route == None or len(route) == 0
    :param to_lane: the goal lane the route connected to
    :return has_path, route 
    '''
    if from_lane == to_lane:
      return True, [(from_lane, None)]

    get_route: List[Tuple[int, RouteEdgeType]] = []

    has_path = self.check_lanes_has_path(from_lane, to_lane)
    if has_path:
      lane_seq_id = networkx.shortest_path(self.lane_graph, from_lane, to_lane)

      len_route = len(lane_seq_id)
      for sid in range(0, len_route):
        edge_type = None
        if (sid + 1) < len_route:
          edge_type = self.get_route_edge_type(
            lane_seq_id[sid], lane_seq_id[sid+1])
        get_route.append((lane_seq_id[sid], edge_type))

    return has_path, get_route

  def append_route(self, route: List[Tuple[int, RouteEdgeType]], 
                         to_lane: int) -> Tuple[bool, List[Tuple[int, RouteEdgeType]]]:
    '''
    append the route given to_lane
    :param route: the original route
    :param to_lane: the goal lane the route connected to
    :return append_success, route 
    '''
    len_input_route = len(route)
    if len_input_route < 1:
      return False, route

    get_route = route
    from_lane = route[-1][0] # using last one as from_lane
    if from_lane == to_lane:
      return True, route

    append_success = self.check_lanes_has_path(from_lane, to_lane)
    if append_success:
      lane_seq_id = networkx.shortest_path(self.lane_graph, from_lane, to_lane)

      len_route = len(lane_seq_id)
      for sid in range(0, len_route):
        edge_type = None
        if (sid + 1) < len_route:
          edge_type = self.get_route_edge_type(
            lane_seq_id[sid], lane_seq_id[sid+1])

        if len(get_route) == len_input_route:
          # first node: modify the original route
          get_route[-1][1] = edge_type
        else:
          # other node: append the original route
          get_route.append((lane_seq_id[sid], edge_type))

    return append_success, get_route

  def extract_frenet_path(self, lane_seq: List[Tuple[int, RouteEdgeType]]) -> Tuple[bool, FrenetPath]:
    '''
    Extract frenet path given a sequence of lane
    :param lane_seq: a sequence of lane id with edge type
    return is_truncated, FrenetPath
      :is_truncated: lane_seq is truncated due to the RouteEdgeType
      :FrenetPath: the frenet being returned
    '''
    mid_points = np.array([]).reshape(0, 3)
    left_points = np.array([]).reshape(0, 3)
    right_points = np.array([]).reshape(0, 3)

    last_lane_idx = -1
    for node in lane_seq:
      lane_idx: int= node[0]
      to_next_lane_edge: RouteEdgeType= node[1]

      # TODO(abing): left/right bound...
      mid_points = np.concatenate(
        (mid_points, self.lane_mid_points[lane_idx]), axis=0)
      left_points = np.concatenate(
        (left_points, self.lane_left_points[lane_idx]), axis=0)
      right_points = np.concatenate(
        (right_points, self.lane_right_points[lane_idx]), axis=0)
      if to_next_lane_edge != RouteEdgeType.FOLLOW:
        # break when LEFT_CHANGE/RIGHT_CHANGE/None
        last_lane_idx = lane_idx
        break

    #:param is_truncated == False: indicates extracted frenet_path 
    #   reaches the last lane of the lane_seq
    route_end_lane_idx = lane_seq[-1][0]
    is_truncated = not (
      (last_lane_idx == route_end_lane_idx) or
      (last_lane_idx in self.lane_edges[route_end_lane_idx]['left']) or
      (last_lane_idx in self.lane_edges[route_end_lane_idx]['right'])
    )

    frenet_path = FrenetPath(mid_points, left_points, right_points)

    # print("frenet_path points", path_points.shape)
    # print("frenet_path sum_dist", frenet_path.max_sum_s())

    return is_truncated, frenet_path

  def get_lane_neighbours(self, check_lane: int) -> Dict[str, List]:
    '''
    Return lane neighbours (@note for left/right neighbour, it may have oppoisite lane direction)
    '''
    if check_lane in self.lane_edges.keys():
      return self.lane_edges[check_lane]
    else:
      return {'pred': [], 'succ': [], 'left': [], 'right': []}

  def get_route_edge_type(self, lane_from: int, lane_to: int, 
                                assert_connectivity: bool = True) -> RouteEdgeType:
    '''
    Return edge type to connect lane_from and lane_to
    :param lane_from: lane index of lane from
    :param lane_to: lane index of lane to
    :param assert_connectivity: if true, will cause error when there is
           not edge between lane_from and lane_to
    '''
    if lane_to in self.lane_edges[lane_from]['succ']:
      return RouteEdgeType.FOLLOW
    if lane_to in self.lane_edges[lane_from]['left']:
      return RouteEdgeType.LEFT_CHANGE
    if lane_to in self.lane_edges[lane_from]['right']:
      return RouteEdgeType.RIGHT_CHANGE
    if assert_connectivity:
      raise ValueError(f"there is no edge from={lane_from} to={lane_to}")
    return None

  def check_left_right_lane_is_same_direction(self, 
      from_lane: int, to_lane: int, edge_key: str) -> bool:
    '''
    Check weather two lane are with same direction
    :param from_lane/to_lane: the two lanes
    :param edge_key: the edge type from the from_lane to the to_lane, its value should be inside ['left' 'right']
    return true when is same direction
    '''
    if not edge_key in ['left', 'right']:
      raise ValueError("Input edge type error.")
    is_same_direct = not (from_lane in self.lane_edges[to_lane][edge_key])
    return is_same_direct

  @staticmethod
  def update_yaw_values(xy_array: np.ndarray) -> np.ndarray:
    '''
    Heuristically update yaw values given xy_array
    : xy_array: xy array with shape = [node_num, >=2: [x, y, ...]]
    '''
    yaws = np.arctan2(xy_array[1:, 1] - xy_array[:-1, 1], 
                      xy_array[1:, 0] - xy_array[:-1, 0])
    yaws = yaws.tolist() + [yaws[-1]]
    xyyaw_array = [[point[0], point[1], yaw] for point, yaw in zip(xy_array, yaws)]
    xyyaw_array = np.array(xyyaw_array)
    return xyyaw_array

  def plot_lane_graph(self, save_path: str=None, fig_dpi: int=300):
    '''
    Plot the lane graph or save it to save_path when save_path != None.
    :param save_path: the path to save the figure, default = None.
    '''
    edg_colors = []
    edg_widths = []
    
    path_edges = self.route_edges
    for u,v in self.lane_graph.edges():
      if (u, v) in path_edges:
        edg_colors.append('r')
        edg_widths.append(1.0)
      else:
        edg_colors.append(self.lane_graph[u][v]['color'])
        edg_widths.append(0.5)

    plt.clf()
    networkx.draw(self.lane_graph, 
                  pos=self.lane_graph_node_xys,
                  node_color='lightgreen', 
                  node_size=2.0,
                  arrowsize=3.5,
                  edge_color=edg_colors,
                  width=edg_widths)

    if save_path:
      plt.savefig(save_path, dpi=fig_dpi)
    else:
      plt.show()
