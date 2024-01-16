from typing import List, Dict, Union, Tuple, Any
import math
import copy
from shapely.geometry import Point, Polygon
from type_utils.aabb_box import AABBBox, PolygonList
from utils.transform import XYYawTransform
from utils.angle_operation import get_normalized_angle, get_normalized_angles
import numpy as np

class TrajectoryInfo:
  def __init__(self,
               scene_id: int = 0,
               agent_type: int = 0,
               length: float = 0.0,
               width: float = 0.0,
               first_time_stamp_s: float = 0.0,
               time_interval_s: float = 0.1
  ):
    '''
    :param scene_id: scene index
    :param agent_type: agent int type
    :param length/width: agent length, width
    :param first_time_stamp_s: first state's time stamp in trajectory (seconds)
    :param time_interval_s: time interval between each states
    '''
    self.scene_id: int = scene_id
    self.agent_type: int = agent_type
    self.length: float = length
    self.width: float = width
    self.first_time_stamp_s: float = first_time_stamp_s
    self.time_interval_s: float = time_interval_s

  def get_info_list(self) -> List:
    return [self.scene_id, self.agent_type,
            self.length, self.width, 
            self.first_time_stamp_s, self.time_interval_s]

  def get_shape_area(self) -> float:
    return (self.length * self.width)

  def debug_info(self) -> str:
    return "TrajectoryInfo= scene_id:{}; agent_type:{}; length/width=({:.2f}/{:.2f});"\
           " time0_s={:.2f}; interval_s={:.2f};"\
      .format(self.scene_id, self.agent_type, self.length, self.width, 
              self.first_time_stamp_s, self.time_interval_s)

class StateTrajectory:
  def __init__(self, info: TrajectoryInfo=None):
    '''
    Trajectory list obey the following rules:
      traj[0] = TrajectoryInfo.get_info_list()
      traj[1:] = [x, y, yaw, signed_velocity, time_seconds, ...]
        where ... represents the array length should in line with traj[0]
    @note the x,y,yaw should be center of agent.
    '''
    self.state_format: Dict[str, int] = {
      'pos_x': 0,
      'pos_y': 1,
      'pos_yaw': 2,
      'velocity': 3,
      'timestamp': 4,
    }

    self.info = None
    self.traj_list = []
    if info:
      self.info = info
      self.traj_list = [info.get_info_list()]
  
  ### Readers
  def get_info(self) -> TrajectoryInfo:
    '''
    Get trajectory info of this trajectory.
    ''' 
    return self.info

  def len(self) -> int:
    return (len(self.traj_list) - 1)

  def duration_s(self) -> float:
    return self.len() * self.info.time_interval_s

  def first_time_stamp_s(self) -> float:
    return self.info.first_time_stamp_s

  def end_time_stamp_s(self) -> float:
    assert len(self.traj_list) > 1, "Error, length of trajectory state list is 0."
    return self.traj_list[-1][self.state_format['timestamp']]

  def debug_info(self) -> str:
    return self.info.debug_info()

  @staticmethod
  def start_state_index() -> int:
    return 1

  def list_trajectory(self) -> List:
    return self.traj_list

  def numpy_trajecotry(self) -> np.ndarray:
    #@note dtype=np.float64 is to support np.sqrt calculation
    return np.array(self.traj_list, dtype=np.float64)

  def numpy_xy_array(self) -> np.ndarray:
    #@note return array abandons trajectory info
    return np.array(self.traj_list, dtype=np.float64)[1:, 0:2]

  def numpy_xyyaw_array(self) -> np.ndarray:
    #@note return array abandons trajectory info
    return np.array(self.traj_list, dtype=np.float64)[1:, 0:3]

  def state_value(self, frame_index:int, key: str) -> float:
    '''
    Check state value of trajectory
    :param frame_index: time frame index in trajectory
    :param key: key tring for looking up, like 'pos_x', 'pos_y' and so on
    :return value of state in trajectory
    '''
    if not key in self.state_format.keys():
      raise ValueError("key={} is not in {}".format(self.state_format.keys()))

    return self.traj_list[1+frame_index][self.state_format[key]]

  def frame_index(self, array_traj: np.ndarray, timestamp: float) -> int:
    '''
    Get frame index of timestamp based on array like trajectory
    '''
    timestamp = round(timestamp, 1) # 1.0xxx > 1.0
    frame_idx = np.where(
      array_traj[1:, self.state_format['timestamp']] > (timestamp-1e-3))[0][0]
    return frame_idx

  def get_local_frame_trajectory(self, origin_xyyaw: Tuple[float, float, float]= None):
    '''
    Return local frame representation trajectory, start is [x=0, y=0, yaw=0]
    :param origin_xyyaw, local trajectory relative to origin_xyyaw, where yaw in radian.
    '''
    get_info = TrajectoryInfo(
      scene_id=self.info.scene_id,
      agent_type=self.info.agent_type,
      length=self.info.length,
      width=self.info.width,
      first_time_stamp_s=self.info.first_time_stamp_s,
      time_interval_s=self.info.time_interval_s
    )
    
    state0 = self.numpy_trajecotry()[1, :]
    xyyaw0 = None
    if isinstance(origin_xyyaw, type(None)):
      xyyaw0 = XYYawTransform(
        x=state0[self.state_format['pos_x']], 
        y=state0[self.state_format['pos_y']], 
        yaw_radian=state0[self.state_format['pos_yaw']])
    else:
      xyyaw0 = XYYawTransform(
        x=origin_xyyaw[0], y=origin_xyyaw[1], 
        yaw_radian=origin_xyyaw[2])
    inv_xyyaw0 = copy.copy(xyyaw0)
    inv_xyyaw0.inverse()

    local_traj = StateTrajectory(get_info)
    for dt in self.numpy_trajecotry()[1:, :]:
      x = dt[self.state_format['pos_x']]
      y = dt[self.state_format['pos_y']]
      yaw = dt[self.state_format['pos_yaw']]
      velocity = dt[self.state_format['velocity']]
      timestamp = dt[self.state_format['timestamp']]

      xyyaw = XYYawTransform(
        x=x, y=y, yaw_radian=yaw)
      start2xyyaw = inv_xyyaw0.multiply_from_right(xyyaw)

      local_traj.append_state(
        start2xyyaw._x, start2xyyaw._y, start2xyyaw._yaw, 
        velocity, timestamp)

    return local_traj

  def split_trajectory(self, split_dt: float, split_dur: float) -> List:
    '''
    Split ego trajectory and return piecewise trajectories 
    :param split_dt, delta time to extract trajectory
    :param split_dur, splited trajectory durations
    '''
    delta_index = round(split_dt / self.info.time_interval_s)
    dur_len = round(split_dur / self.info.time_interval_s)

    len_traj_nodes = len(self.traj_list) - 1

    if len_traj_nodes < dur_len:
      indexs_from = [0]
    else:
      indexs_from = [i for i in range(0, len_traj_nodes, delta_index) if (i+dur_len) < len_traj_nodes]

    get_trajs = []
    traj_body = self.numpy_trajecotry()[1:, :]
    for index_from in indexs_from:
      frames = list(range(index_from, min(index_from+dur_len, len_traj_nodes)))
      piece_traj_body = traj_body[frames]

      get_info = TrajectoryInfo(
        scene_id=self.info.scene_id,
        agent_type=self.info.agent_type,
        length=self.info.length,
        width=self.info.width,
        first_time_stamp_s=self.info.first_time_stamp_s,
        time_interval_s=self.info.time_interval_s
      )
      get_info.first_time_stamp_s = piece_traj_body[0, self.state_format['timestamp']]
      
      piece_traj = StateTrajectory(get_info)
      piece_traj.traj_list += piece_traj_body.tolist()
      get_trajs.append(piece_traj)

    return get_trajs

  ### Setters
  def set_trajectory_list(self, traj_list: List) -> None:
      if len(traj_list) >= 1:
        cache = traj_list[0]
        info = TrajectoryInfo(
          scene_id=int(cache[0]),
          agent_type=int(cache[1]),
          length=cache[2], width=cache[3],
          first_time_stamp_s=cache[4],
          time_interval_s=cache[5],
        )

        self.info = info
        self.traj_list = traj_list

  def set_state_value(self, frame_index:int, key: str, value: Union[int, float]) -> None:
    '''
    Set value of specific state
    '''
    self.traj_list[1+frame_index][self.state_format[key]] = value

  def append_state(self, x: float, y: float, yaw: float, 
                         signed_velocity: float,
                         relative_tstamp: float) -> bool:
    '''
    Append a trajectory state to self.traj_list
    : return True if append successfully.
    '''
    if len(self.traj_list) >= 1:
      # length should in line with trajectory info
      self.traj_list.append([x, y, yaw, signed_velocity, relative_tstamp, 0.])
      return True
    else:
      return False

  ### Special processes
  def get_merged_polys(self, frame_from: int, frame_to: int, 
                             merged_interval: int,
                             inflation_length: float=0.2,
                             inflation_width: float=0.2) -> PolygonList:
    '''
    Extract aabb boxes given a trajectory and its info (shape extents)
    :param frame_from/frame_to: indicates from/to which frame to extract
    :param merged_interval: indicates between how many frames to generate a aabbbox
            where, [i, i+merged_interval-1] states are picked up to generete the box
    :param inflation_length/width: inflation of agent length/width
    return list of AABBBox with information:
            [{'frame_from': int, 'frame_to': int,
              'time_begin_s': float, 'time_end_s': float, 
              'aabbbox': AABBBox}, ...]
    '''
    ori_traj_len = len(self.traj_list)
    index_list = range(1+frame_from, 1+frame_to+1, merged_interval)

    abs_dx = self.info.length * 0.5
    abs_dy = self.info.width * 0.5 + inflation_width
    dposes = []
    for dx in [-abs_dx, abs_dx + inflation_length]:
      for dy in [-abs_dy, abs_dy]:
        dposes.append(XYYawTransform(x=dx, y=dy))

    boxes_list = PolygonList()
    boxes_list.poly_list = []

    wx_min, wx_max, wy_min, wy_max = +np.inf, -np.inf, +np.inf, -np.inf
    for index_from in index_list:
      index_to = min(index_from+merged_interval, ori_traj_len)
      frame_from = index_from - 1
      frame_to = index_to - 1

      state_lists = self.traj_list[index_from: index_to]
      if len(state_lists) == 0:
        # warnings.warn(message='Unexpected state list len == 0')
        # print("details=", index_from, index_to, ori_traj_len)
        continue # skip invalid state list
      elif len(state_lists) == 1:
        state_lists = [state_lists[0]]
      else:
        state_lists = [state_lists[0], state_lists[-1]]

      state_x_idx:int = self.state_format['pos_x']
      state_y_idx:int = self.state_format['pos_y']
      state_yaw_idx:int = self.state_format['pos_yaw']
      state_xyyaw_idxs = [state_x_idx, state_y_idx, state_yaw_idx]

      x_min, x_max, y_min, y_max = +np.inf, -np.inf, +np.inf, -np.inf
      list_xy_points = []
      for state in state_lists:
        state_x = state[state_x_idx]
        state_y = state[state_y_idx]
        state_yaw = state[state_yaw_idx]
        xyyaw = XYYawTransform(x=state_x, y=state_y, yaw_radian=state_yaw)
        for dpose in dposes:
          corner_point = xyyaw.multiply_from_right(dpose)
          x_min = min(x_min, corner_point._x)
          x_max = max(x_max, corner_point._x)
          y_min = min(y_min, corner_point._y)
          y_max = max(y_max, corner_point._y)
          list_xy_points.append((corner_point._x, corner_point._y))
      wx_min = min(wx_min, x_min)
      wx_max = max(wx_max, x_max)
      wy_min = min(wy_min, y_min)
      wy_max = max(wy_max, y_max)

      state_b = [state_lists[0][j] for j in state_xyyaw_idxs]
      state_e = [state_lists[-1][j] for j in state_xyyaw_idxs]
      dx = state_e[0] - state_b[0]
      dy = state_e[1] - state_b[1]
      move_direction = state_b[2]
      move_dist = math.sqrt(dx*dx + dy*dy)
      if (len(state_lists) > 1) and (move_dist > 0.25):
        move_direction = math.atan2(dy, dx)

      convex_poly = Polygon(tuple(list_xy_points)).convex_hull

      mid_x = (state_b[0] + state_e[0]) * 0.5
      mid_y = (state_b[1] + state_e[1]) * 0.5
      mid_dist = math.sqrt((state_b[0] - mid_x)**2 + (state_b[1] - mid_y)**2) * 0.5
      circle = Point(mid_x, mid_y).buffer(mid_dist + abs_dy)

      time_begin_s = state_lists[0][self.state_format['timestamp']]
      time_end_s = state_lists[-1][self.state_format['timestamp']]
      boxes_list.poly_list.append({
          'frame_from': frame_from,
          'frame_to': frame_to,
          'time_begin_s': time_begin_s,
          'time_end_s': time_end_s,
          'move_direction': move_direction,
          'aabbbox': AABBBox([x_min, x_max], [y_min, y_max]),
          'convex_poly': convex_poly,
          'circle': circle,
        }
      )

    boxes_list.aabb = AABBBox([wx_min, wx_max], [wy_min, wy_max])
    return boxes_list

  @staticmethod
  def check_overlap(poly_list1: PolygonList,
                    poly_list2: PolygonList,
                    dist_time_overlap: float=0.01,
                    check_clearance: bool=False) -> Dict:
    '''
    Return overlap information of two box list
    :param box_list: folowing the formats in get_merged_boxes(), where is a list of 
                     boxes with time extent.
    :param check_clearance: if set true, the clearance between polygons are calculated
    '''
    st_overlapped: bool = False
    min_clearance: float = 1e+3
    clearance_dict: Dict[int, float] = {} # map: idx: clearance

    quick_skip = False
    if (check_clearance == False):
      quick_skip = not AABBBox.overlapped(poly_list1.aabb, poly_list2.aabb)

    if quick_skip == False:
      for bid1, binfo1 in enumerate(poly_list1.poly_list):
        t_l1 = binfo1['time_begin_s']
        t_u1 = binfo1['time_end_s']
        direct1 = binfo1['move_direction']
        box1 = binfo1['aabbbox']
        poly1 = binfo1['convex_poly']

        min_dist = 1e+3
        for bid2, binfo2 in enumerate(poly_list2.poly_list):
          t_l2 = binfo2['time_begin_s']
          t_u2 = binfo2['time_end_s']
          direct2 = binfo2['move_direction']
          box2 = binfo2['aabbbox']
          poly2 = binfo2['convex_poly']

          time_overlapped = not (((t_u1 + dist_time_overlap) < t_l2) or
                                ((t_l1 - dist_time_overlap) > t_u2))
          if time_overlapped and check_clearance:
            get_dist = poly1.distance(poly2)
            if get_dist < min_dist:
              min_dist = get_dist

          # check time overlapped
          if time_overlapped:
            # check aabb is overlapped
            if AABBBox.overlapped(box1, box2):
              # accurately check polygon overlapped
              if poly1.overlaps(poly2):
                st_overlapped = True

        if check_clearance:
          min_clearance = min(min_clearance, min_dist)
          clearance_dict[bid1] = min_dist

        if (st_overlapped):
          break

    return {
      'st_overlapped': st_overlapped,

      'clearance_dict': clearance_dict,
      'min_clearance': min_clearance,
    }

  def check_space_interaction(self,
                              compare_traj, # <StateTrajectory>
                              point_width: float=0.2) -> Dict:
    idx_x:int = self.state_format['pos_x']
    idx_y:int = self.state_format['pos_y']
    idx_yaw:int = self.state_format['pos_yaw']
    idx_v:int = self.state_format['velocity']
    idx_t:int = self.state_format['timestamp']
    
    array1 = self.numpy_trajecotry()[1:, :]
    array2 = compare_traj.numpy_trajecotry()[1:, :]
    array1_xys = array1[:, [idx_x, idx_y]]
    array2_xys = array2[:, [idx_x, idx_y]]
    
    array1_piecewise_dists = np.linalg.norm(array1_xys[1:, :] - array1_xys[:-1, :], axis=1)
    array2_piecewise_dists = np.linalg.norm(array2_xys[1:, :] - array2_xys[:-1, :], axis=1)

    # ################################################################
    point_interaction = None
    min_overlap_dist:float = +np.inf
    shift_dist:float = point_width * 0.25

    has_overlap :bool= False
    space_interaction = {
      'has_overtake': False,
      'has_giveway': False,
      'has_both_overtake_and_giveway': False,
      'overlap_tags': [],
      'overlap_ts': [],
    }

    for iid, tpi in enumerate(array1):
      # print("t-xy-v[{:.1f},{:.1f},{:.1f},{:.1f}]".format(
      #   tpi[idx_t], tpi[idx_x], tpi[idx_y], tpi[idx_v]))
      dxys = array2_xys - tpi[[idx_x, idx_y]]
      dists = np.linalg.norm(dxys, axis=1)

      # data: space interaction
      min_dist = np.min(dists)
      if (min_dist < point_width):
        jid = np.argmin(dists)
        tpi_t = tpi[idx_t]
        tpj_t = array2[jid, idx_t]
        dt_value = (tpi_t - tpj_t)
        dt_tag = (dt_value >= 0.0) * 2.0 - 1.0

        space_interaction['overlap_tags'].append(dt_tag) # -1.0: overtake, 1.0: giveway
        space_interaction['overlap_ts'].append([tpi_t, tpj_t, dt_value])
      else:
        space_interaction['overlap_tags'].append(0.0)
        space_interaction['overlap_ts'].append([0., 0., 0.])
      
      # data: point_interaction
      if (min_dist < min_overlap_dist) and (min_dist < point_width):
        jid2 :int= np.where(dists < min(min_overlap_dist, point_width))[0][0]
        tpi_t = tpi[idx_t]
        tpi_yaw = tpi[idx_yaw]

        tpj_t = array2[jid2, idx_t]
        tpj_yaw = array2[jid2, idx_yaw]

        point_interaction = {
          'ipoint_index': [iid, jid2], 
          'arrive_t_s': [tpi_t, tpj_t],
          'arrive_dt_s': tpi_t - tpj_t,
          'directs': [tpi_yaw, tpj_yaw],
          'overlap_angle': get_normalized_angle(tpi_yaw - tpj_yaw),
        }
        min_overlap_dist = min_dist - shift_dist
        has_overlap = True

    space_interaction['overlap_tags'] = np.array(space_interaction['overlap_tags'])
    space_interaction['overlap_ts'] = np.array(space_interaction['overlap_ts'])
    space_interaction['has_overtake'] = np.sum(
      (space_interaction['overlap_tags'] < -1e-3) * 1.0) > 1e-3 # overlap_tags < 0.0 -> overtakes
    space_interaction['has_giveway'] = np.sum(
      (space_interaction['overlap_tags'] > 1e-3) * 1.0) > 1e-3  # overlap_tags > 0.0 -> giveways
    space_interaction['has_both_overtake_and_giveway'] =\
      space_interaction['has_overtake'] * space_interaction['has_giveway']

    interact_info = {
      'piecewise_dists': [array1_piecewise_dists, array2_piecewise_dists],
      'has_overlap': has_overlap,
      'space_interaction': space_interaction,
      'point_interaction': point_interaction,
    }

    return interact_info
