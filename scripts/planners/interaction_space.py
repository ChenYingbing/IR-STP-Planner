from typing import Any, Dict, List, Union

from scipy.spatial import KDTree
import numpy as np
import math
import abc
from typing import Tuple
from shapely.geometry import Polygon, MultiPolygon

from utils.angle_operation import get_normalized_angle, get_normalized_angles
from pymath.spaceXd import SpaceXd
import matplotlib.pyplot as plt

TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE
M_PI = 3.1416

class InteractionFormat:
  def __init__(self, plan_horizon_T: float, 
               plan_dt: float=0.1, prediction_dt: float=0.5) -> None:
    '''
    Format the interaction information
    :param plan_horizon_T: planning time horizone
    :param plan_dt: planning node interval timestamp
    :param prediction_dt: prediction trajectory node interval timestamp
    '''
    self.prediction_dt = prediction_dt
    self.half_prediction_dt = prediction_dt * 0.5

    self.cache2keys = {
      'path_x': int(0),         # path pose x for the AV
      'path_y': int(1),         # path pose y for the AV
      'path_yaw': int(2),       # path pose yaw for the AV
      'pred_idx': int(3),       # prediction agent index
      'pred_initial_v': int(4), # prediction agent's initial speed 
      'pred_traj_idx': int(5),       # prediction trajectory index
      'pred_traj_prob': int(6),      # prediction trajectory probability
      'pred_traj_point_s': int(7),   # overlapped point s value of the prediction trajectory
      'pred_traj_point_t': int(8),   # overlapped point t value of the prediction trajectory
      'pred_traj_point_yaw': int(9), # overlapped point yaw value of the prediction trajectory
      'pred_traj_point_v': int(10),  # overlapped point speed value of the prediction trajectory
      'pred_traj_acc': int(11) # agent's acc value (perception result)
    }

  def get_key_idx(self, key_str: str) -> int:
    '''
    Return corresponding data index given string, its value follows self.cache2keys.keys():
      ['path_x/y/yaw', 'pred_idx', 'pred_traj_idx', 'pred_traj_prob', 
       'pred_traj_point_t', 'pred_traj_point_yaw', 'pred_traj_point_v',
       'pred_traj_acc']
    '''
    return self.cache2keys[key_str]

  def data_num(self) -> int:
    return len(self.cache2keys)

  def get_list_of_data(self, path_pose: np.ndarray, 
                       pred_idx: int, pred_initial_v: float,
                       pred_traj_idx: int, pred_traj_prob: float,
                       pred_traj_point_s: float, pred_traj_point_t: float, 
                       pred_traj_point_yaw: float, pred_traj_point_v: float,
                       pred_traj_acc: float) -> List:
    '''
    Return data in the list format
    :param path_pose: pose of the path of the AV, in format: [x, y, yaw]
    :param pred_idx: prediction index (agent)
    :param pred_initial_v: prediction agent's initial speed
    :param pred_traj_idx: prediction trajectory index
    :param pred_traj_prob: trajectory probability
    :param pred_traj_point_s: sum of longitidinal distance: s when agent reaches this interaction space
    :param pred_traj_point_t: time when agent reaches this interaction space
    :param pred_traj_point_yaw: yaw state when agent reaches this interaction space
    :param pred_traj_point_v: speed value when agent reaches this interaction space
    '''
    return_list = [path_pose[0], path_pose[1], path_pose[2], 
                   pred_idx, pred_initial_v, pred_traj_idx, pred_traj_prob, 
                   pred_traj_point_s, pred_traj_point_t, pred_traj_point_yaw, 
                   pred_traj_point_v, pred_traj_acc]
    assert len(return_list) == len(self.cache2keys), "Fatal error, size unmatched."

    return return_list

  def add_iinfo2dict(self, path_s: float, array_data: np.ndarray, input_dict: Dict) -> None:
    '''
    Add interaction inforamtion to the input_dict variable
    :param path_s: path s value
    :param array_data: raw input interaction data related to this path point (s == path_s), 
            the shape is (n, len(get_list_of_data()))
    :param input_dict: dict to record data, format follows:
      's/id_trajid_v0_s_t_v_acc_iangle': record path_point_s, agent_v0, agent_index, agent_traj_index
          agent_s_to_path_point, agent_t_to_path_point, agent_v_at_path_point, interaction_angle
    '''
    for row_dt in array_data:
      # av's yaw angle at this path point
      path_yaw = row_dt[self.cache2keys['path_yaw']]
      # agent index
      agent_idx = int(row_dt[self.cache2keys['pred_idx']])
      # agent prediction trajectory index
      agent_traj_idx = int(row_dt[self.cache2keys['pred_traj_idx']])
      # agent initial speed at this prediction
      agent_v0 = row_dt[self.cache2keys['pred_initial_v']]
      # agent_stv: agent s, t ,v values along predictions
      agent_s = row_dt[self.cache2keys['pred_traj_point_s']]
      agent_t = row_dt[self.cache2keys['pred_traj_point_t']]
      agent_v = row_dt[self.cache2keys['pred_traj_point_v']]
      agent_acc = row_dt[self.cache2keys['pred_traj_acc']]
      # agent's yaw angle when arriving this path point
      agent_yaw = row_dt[self.cache2keys['pred_traj_point_yaw']]

      # interaction angle
      iangle_signed = get_normalized_angle(path_yaw - agent_yaw)

      # init the dict
      if not agent_idx in input_dict:
        input_dict[agent_idx] = {}
      if not agent_traj_idx in input_dict[agent_idx]:
        input_dict[agent_idx][agent_traj_idx] = {'s/id_trajid_v0_s_t_v_acc_iangle': []}

      # add information to input_dict, the format are as following:
      input_dict[agent_idx][agent_traj_idx]['s/id_trajid_v0_s_t_v_acc_iangle'].append(
        [path_s, agent_idx, agent_traj_idx, agent_v0, agent_s, agent_t, 
         agent_v, agent_acc, iangle_signed])

  @staticmethod
  def iformat_len() -> int:
    '''
    Return data num of the add_iinfo2dict() outputs
    '''
    return 10
  
  @staticmethod
  def iformat_index(key_str: str) -> int:
    '''
    Return index of corresponding data for add_iinfo2dict() outputs
    '''
    format_map = {
      'av_s': 0,
      'agent_idx': 1,
      'agent_traj_idx': 2,
      'agent_v0': 3,
      'agent_s': 4,
      'agent_t': 5,
      'agent_v': 6,
      'agent_acc': 7,
      'iangle': 8
    }
    return format_map[key_str]

class InteractionSpace:
  def __init__(self, plan_horizon_T: float,
                     space_range_x: List,
                     space_range_y: List, 
                     space_reso: float,
                     yaw_reso: float,
                     planning_dt: float,
                     prediction_dt: float) -> None:
    '''
    :param plan_horizon_T: plan horizon T
    :param space_range_x: space range in x, [x_min, x_max]
    :param space_range_y: space range in y, [y_min, y_max]
    :param space_reso: space resolution (meters) for discretizing the search space
    :param yaw_reso: yaw resolution (radian) for discretizing the search space
    :param planning_dt: interval timestamp in planning
    :param prediction_dt: time interval among predictions
    '''
    self.plan_horizon_T = plan_horizon_T
    self.prediction_dt: float = prediction_dt
    self.planning_dt = planning_dt

    # map agent_idx > list of prediction trajectory's records
    self.pred_records: Dict[int, List[Dict[str, Any]]]= {}


  def reinit(self, origin_xy: List[float]) -> None:
    '''
    ReInit the interaction space
    :param origin_xy: origin [x, y] of the interaction space
    '''
    self.pred_records.clear()

  def add_predictions(self, agent_idx: int, length: float, width: float,
                            velocity: float, acceleration: float,
                            predictions: List[Union[Dict, np.array]], 
                            inflations: Tuple[float, float] = (0., 0.)) -> None:
    '''
    Add prediction trajectory points and their shape
    :param agent_idx: index of the agent ('pred' mode) / index of the behavior ('plan' mode)
    :param velocity: velocity value of the agent
    :param acceleration: acceleration value of the agent
    :param length: length of the agent
    :param width: width of the agent
    :param predictions: predictions of agent: [{'prob': float, 'trajectory': numpy.array<shape=(pred_num, 3)>}, ...]
    :param inflations: inflation of the (length, width) of the agent
    '''
    half_length = length * 0.5 + inflations[0]
    half_width = width * 0.5 + inflations[1]

    corner_dxys = np.array(
      [[half_length, half_width], [-half_length, half_width], [-half_length, -half_width], [half_length, -half_width]])

    for pred in predictions:
      prob = pred['prob']
      pred_points = pred['trajectory'] # e.g., (13, 3)

      # TODO(abing): imrpove this function if prediction not started from t = 0.0
      tstamps = np.array([i * self.prediction_dt for i, _ in enumerate(pred_points)])

      # update predition records
      if not agent_idx in self.pred_records:
        self.pred_records[agent_idx] = []
      dxys = pred_points[1:, :2] - pred_points[:-1, :2]
      piece_dists = np.linalg.norm(dxys, axis=1)
      sum_s_list = np.zeros([1])
      sum_s_list = np.hstack((sum_s_list, piece_dists))
      for i in range(1, len(sum_s_list)):
        sum_s_list[i] = sum_s_list[i-1] + sum_s_list[i]
      speeds = (sum_s_list[1:] - sum_s_list[:-1]) / self.prediction_dt
      speeds = np.hstack((speeds, np.array([speeds[-1]])))

      points_sin = np.sin(pred_points[:, 2])
      points_cos = np.cos(pred_points[:, 2])
      corner0_xs = corner_dxys[0, 0] * points_cos - corner_dxys[0, 1] * points_sin + pred_points[:, 0]
      corner1_xs = corner_dxys[1, 0] * points_cos - corner_dxys[1, 1] * points_sin + pred_points[:, 0]
      corner2_xs = corner_dxys[2, 0] * points_cos - corner_dxys[2, 1] * points_sin + pred_points[:, 0]
      corner3_xs = corner_dxys[3, 0] * points_cos - corner_dxys[3, 1] * points_sin + pred_points[:, 0]
      
      corner0_ys = corner_dxys[0, 0] * points_sin + corner_dxys[0, 1] * points_cos + pred_points[:, 1]
      corner1_ys = corner_dxys[1, 0] * points_sin + corner_dxys[1, 1] * points_cos + pred_points[:, 1]
      corner2_ys = corner_dxys[2, 0] * points_sin + corner_dxys[2, 1] * points_cos + pred_points[:, 1]
      corner3_ys = corner_dxys[3, 0] * points_sin + corner_dxys[3, 1] * points_cos + pred_points[:, 1]

      polys = [ Polygon(tuple([[x0, y0], [x1, y1], [x2, y2], [x3, y3]]))\
        for x0, y0, x1, y1, x2, y2, x3, y3 in zip(corner0_xs, corner0_ys, corner1_xs, corner1_ys, corner2_xs, corner2_ys, corner3_xs, corner3_ys)]
      # poly_points = np.array([[x0, y0, x1, y1, x2, y2, x3, y3]
      #   for x0, y0, x1, y1, x2, y2, x3, y3 in zip(corner0_xs, corner0_ys, corner1_xs, corner1_ys, corner2_xs, corner2_ys, corner3_xs, corner3_ys)
      # ])

      self.pred_records[agent_idx].append(
        {
          'prob': prob,
          'points': pred_points,
          'velocity': velocity,
          'acceleration': acceleration,
          'sum_s_list':  sum_s_list,
          'tstamps': tstamps,
          'speeds': speeds,
          'polys': polys,
          # 'poly_points': poly_points,
          'multi_polys': MultiPolygon(polygons=polys),
        }
      )

  def read_interactions(self, ego_length: float, ego_width: float, 
      front_inflation: float, behavior_xyyaws: np.ndarray) -> Tuple:
    '''
    Read interaction information of the position of behavioral xy array
    :param ego_length: length of the ego agent
    :param ego_width: width of the ego agent
    :param front_inflation: inflation of the front length
    :param behavior_xyyaws: [[x, y, yaw], ...] with shape (N, 3)
    '''
    length_2 = ego_length * 0.5
    front_length_2 = length_2 + front_inflation
    width_2 = ego_width * 0.5
    points_sin = np.sin(behavior_xyyaws[:, 2])
    points_cos = np.cos(behavior_xyyaws[:, 2])

    # for av already occupy relation, using a more conservative shape
    init_poly_l0 = -0.5
    init_poly_l1 = 0.5
    init_poly_l2 = length_2 - 0.5
    ego_init_yaw = None
    ego_rear_init_poly = None
    ego_front_init_poly = None
    if behavior_xyyaws.shape[0] > 0:
      ego_init_yaw = behavior_xyyaws[0, 2]
      cdxys = np.array(
        [[init_poly_l0, width_2], [-length_2, width_2], 
         [-length_2, -width_2], [init_poly_l0, -width_2]])
      init_c0_x = cdxys[0, 0] * points_cos[0] - cdxys[0, 1] * points_sin[0] + behavior_xyyaws[0, 0]
      init_c1_x = cdxys[1, 0] * points_cos[0] - cdxys[1, 1] * points_sin[0] + behavior_xyyaws[0, 0]
      init_c2_x = cdxys[2, 0] * points_cos[0] - cdxys[2, 1] * points_sin[0] + behavior_xyyaws[0, 0]
      init_c3_x = cdxys[3, 0] * points_cos[0] - cdxys[3, 1] * points_sin[0] + behavior_xyyaws[0, 0]
      init_c0_y = cdxys[0, 0] * points_sin[0] + cdxys[0, 1] * points_cos[0] + behavior_xyyaws[0, 1]
      init_c1_y = cdxys[1, 0] * points_sin[0] + cdxys[1, 1] * points_cos[0] + behavior_xyyaws[0, 1]
      init_c2_y = cdxys[2, 0] * points_sin[0] + cdxys[2, 1] * points_cos[0] + behavior_xyyaws[0, 1]
      init_c3_y = cdxys[3, 0] * points_sin[0] + cdxys[3, 1] * points_cos[0] + behavior_xyyaws[0, 1]
      ego_rear_init_poly = Polygon(tuple([
        [init_c0_x, init_c0_y], [init_c1_x, init_c1_y], 
        [init_c2_x, init_c2_y], [init_c3_x, init_c3_y]]))

      cdxys = np.array(
        [[init_poly_l2, width_2], [init_poly_l1, width_2], 
         [init_poly_l1, -width_2], [init_poly_l2, -width_2]])
      init_c0_x = cdxys[0, 0] * points_cos[0] - cdxys[0, 1] * points_sin[0] + behavior_xyyaws[0, 0]
      init_c1_x = cdxys[1, 0] * points_cos[0] - cdxys[1, 1] * points_sin[0] + behavior_xyyaws[0, 0]
      init_c2_x = cdxys[2, 0] * points_cos[0] - cdxys[2, 1] * points_sin[0] + behavior_xyyaws[0, 0]
      init_c3_x = cdxys[3, 0] * points_cos[0] - cdxys[3, 1] * points_sin[0] + behavior_xyyaws[0, 0]
      init_c0_y = cdxys[0, 0] * points_sin[0] + cdxys[0, 1] * points_cos[0] + behavior_xyyaws[0, 1]
      init_c1_y = cdxys[1, 0] * points_sin[0] + cdxys[1, 1] * points_cos[0] + behavior_xyyaws[0, 1]
      init_c2_y = cdxys[2, 0] * points_sin[0] + cdxys[2, 1] * points_cos[0] + behavior_xyyaws[0, 1]
      init_c3_y = cdxys[3, 0] * points_sin[0] + cdxys[3, 1] * points_cos[0] + behavior_xyyaws[0, 1]
      ego_front_init_poly = Polygon(tuple([
        [init_c0_x, init_c0_y], [init_c1_x, init_c1_y], 
        [init_c2_x, init_c2_y], [init_c3_x, init_c3_y]]))

    # for av overlaps with trajectories, using an inflated shape
    corner_dxys = np.array(
      [[front_length_2, width_2], [-length_2, width_2], [-length_2, -width_2], [front_length_2, -width_2]])
    corner0_xs = corner_dxys[0, 0] * points_cos - corner_dxys[0, 1] * points_sin + behavior_xyyaws[:, 0]
    corner1_xs = corner_dxys[1, 0] * points_cos - corner_dxys[1, 1] * points_sin + behavior_xyyaws[:, 0]
    corner2_xs = corner_dxys[2, 0] * points_cos - corner_dxys[2, 1] * points_sin + behavior_xyyaws[:, 0]
    corner3_xs = corner_dxys[3, 0] * points_cos - corner_dxys[3, 1] * points_sin + behavior_xyyaws[:, 0]

    corner0_ys = corner_dxys[0, 0] * points_sin + corner_dxys[0, 1] * points_cos + behavior_xyyaws[:, 1]
    corner1_ys = corner_dxys[1, 0] * points_sin + corner_dxys[1, 1] * points_cos + behavior_xyyaws[:, 1]
    corner2_ys = corner_dxys[2, 0] * points_sin + corner_dxys[2, 1] * points_cos + behavior_xyyaws[:, 1]
    corner3_ys = corner_dxys[3, 0] * points_sin + corner_dxys[3, 1] * points_cos + behavior_xyyaws[:, 1]

    ego_polys = [ Polygon(tuple([[x0, y0], [x1, y1], [x2, y2], [x3, y3]]))\
      for x0, y0, x1, y1, x2, y2, x3, y3 in zip(corner0_xs, corner0_ys, corner1_xs, corner1_ys, corner2_xs, corner2_ys, corner3_xs, corner3_ys)]
    # self.__rviz_ego_polys = np.array([[x0, y0, x1, y1, x2, y2, x3, y3]\
    #   for x0, y0, x1, y1, x2, y2, x3, y3 in zip(
    #     corner0_xs, corner0_ys, corner1_xs, corner1_ys, corner2_xs, corner2_ys, corner3_xs, corner3_ys)])
    ego_multi_polys = MultiPolygon(ego_polys)

    overlap_cond_dist :float= 1e-1
    overlap_flag = {}
    formatter = InteractionFormat(plan_horizon_T=self.plan_horizon_T)

    result_list = []
    av_rear_already_occupied_dict = {}
    av_front_already_occupied_dict = {}
    for bp_idx, ego_poly in enumerate(ego_polys):
      fill_dt = {'is_interacted': False, 'details': [] }      
      bpoint = behavior_xyyaws[bp_idx, :]

      for agent_idx, list_content in self.pred_records.items():
        for pred_traj_idx, content in enumerate(list_content):
          pred_prob = content['prob']
          agent_polys = content['polys']
          agent_multi_poly = content['multi_polys']
          agent_sum_s_list = content['sum_s_list']
          pred_speed0 = content['velocity']
          agent_acc = content['acceleration']
          pred_points = content['points']

          if not (agent_idx, pred_traj_idx) in overlap_flag.keys():
            overlap_flag[(agent_idx, pred_traj_idx)] =\
              ego_multi_polys.distance(agent_multi_poly) < overlap_cond_dist
          if overlap_flag[(agent_idx, pred_traj_idx)] == False:
            continue # skip this prediction for this behavior_xyyaws as they are not overlapped

          # here, path is overlapped with agent_traj
          point_dist = ego_poly.distance(agent_multi_poly)
          if point_dist < overlap_cond_dist:
            # if point is overlapped with agent_traj
            for pp_idx, agent_poly in enumerate(agent_polys):
              dist = ego_poly.distance(agent_poly)
              # ego_poly.intersection(agent_poly).area()

              if dist < overlap_cond_dist:
                ppoint = pred_points[pp_idx, :]
                ppoint_yaw = ppoint[2]
                tsvalue = agent_sum_s_list[pp_idx]
                tstamp = content['tstamps'][pp_idx]
                tspeed = content['speeds'][pp_idx]

                fill_dt['is_interacted'] = True
                fill_dt['details'].append(
                  formatter.get_list_of_data(
                    path_pose=bpoint,
                    pred_idx=agent_idx,
                    pred_initial_v=pred_speed0,
                    pred_traj_idx=pred_traj_idx,
                    pred_traj_prob=pred_prob,
                    pred_traj_point_s=tsvalue,
                    pred_traj_point_t=tstamp,
                    pred_traj_point_yaw=ppoint_yaw,
                    pred_traj_point_v=tspeed,
                    pred_traj_acc=agent_acc)
                )

                iangle = math.fabs(get_normalized_angle(ego_init_yaw - ppoint_yaw)) * TO_DEGREE
                if (bp_idx == 0) and (iangle <= 155.0): # only not inv interaction can be ignored in relation 
                  check_dist2init0 = ego_rear_init_poly.distance(agent_multi_poly)
                  check_dist2init1 = ego_front_init_poly.distance(agent_multi_poly)
                  if check_dist2init0 < 1e-1:
                    av_rear_already_occupied_dict[(agent_idx, pred_traj_idx)] = True
                  if check_dist2init1 < 1e-1:
                    av_front_already_occupied_dict[(agent_idx, pred_traj_idx)] = True

          # else:
          #   break # skip this prediction for point: bp_idx 
          # CAN NOT BE SKIP, MAY OCCURS MISS av_already_occupied situation

      # fill result_list
      fill_dt['details'] = np.array(fill_dt['details'])
      result_list.append(fill_dt)

    # print("\ndebug overlap_flag")
    # print(overlap_flag)
    return result_list, av_rear_already_occupied_dict, av_front_already_occupied_dict

  def plot_av_path_and_agent_predicitons(self, plot_axis, valid_list: List=[]) -> None:
    '''
    plot planning path and prediction results
    '''
    valid_agent_list = self.pred_records.keys()
    if len(valid_list) > 0:
      valid_agent_list = valid_list

    for agent_idx in valid_agent_list:
      if agent_idx in self.pred_records.keys():

        for traj_idx, content in enumerate(self.pred_records[agent_idx]):
          agent_points :np.ndarray= content['points']
          agent_tstamps :np.ndarray= content['tstamps']

          plot_axis.plot(agent_points[:, 0], agent_points[:, 1], 'k-', lw=0.5)
          plot_axis.scatter(
            agent_points[:, 0], agent_points[:, 1], c=agent_tstamps,
            vmin=0.0, vmax=np.max(agent_tstamps), cmap=plt.cm.get_cmap('jet'), marker='o'
          )

        # print("plot_av_path_and_agent_predicitons::plot agent with index={}.".format(agent_idx))
      else:
        print("plot_av_path_and_agent_predicitons::skip plot agent with index="
              "{}, since it did not occurs".format(agent_idx))

  def plot_av_and_agent_polygons(self, plot_axis, valid_list: List=[]) -> None:
    '''
    plot planning path and prediction results
    '''
    valid_agent_list = self.pred_records.keys()
    if len(valid_list) > 0:
      valid_agent_list = valid_list

    # for idx, points in enumerate(self.__rviz_ego_polys[::5, :]):
    #   plot_axis.plot(points[[0, 2, 4, 6, 0]], points[[1, 3, 5, 7, 1]], 'k-', lw=0.25)

    # for agent_idx in valid_agent_list:
    #   if agent_idx in self.pred_records.keys():
    #     for traj_idx, content in enumerate(self.pred_records[agent_idx]):
    #       agent_points :np.ndarray= content['points']
    #       agent_poly_points :np.ndarray= content['poly_points']
          
    #       if agent_idx == 30015:
    #         for idx, points in enumerate(agent_poly_points):
    #           plot_axis.plot(points[[0, 2, 4, 6, 0]], points[[1, 3, 5, 7, 1]], 'r-', lw=0.25)
    #         plot_axis.plot(agent_points[:, 0], agent_points[:, 1], 'x-', lw=2.0)
    
# def process_interactions(self, safe_clearance: float=0.0):
#   '''
#   Process traffic interactions based on the data of plan_kdtree/pred_kdtrees.
#   :param safe_clearance: inflation or radius for agents (multi-circle model).
#   '''
#   # print("***"*30)
#   formatter = InteractionFormat(plan_horizon_T=self.plan_horizon_T)

#   agent_info_xy_locs: List= []
#   repeat_checker_dict = {}
#   for bidx, btree in self.plan_kdtree.items():
#     #:param bidx: behavior index
#     #:param btree: kdtree of the behavior
#     binfo = self.plan_kdt_dt[bidx]['info']
#     bpoints = self.plan_kdt_dt[bidx]['points']
#     # print("behavior {} with radius={:.1f}, points={}.".format(
#     #   bidx, binfo['radius'], bpoints.shape))
    
#     for pidx, ptree in self.pred_kdtree.items():
#       #:param pidx: prediction index
#       #:param ptree: kdtree of the prediciton
#       pinfo = self.pred_kdt_dt[pidx]['info']
#       probs = self.pred_kdt_dt[pidx]['prob']
#       ppoints = self.pred_kdt_dt[pidx]['points']
#       # print("analyze interactions between b{}/p{}, p_radius={:.1f}, with points={}.".format(
#       #   bidx, pidx, pinfo['radius'], ppoints.shape))

#       pred_traj_len: int = self.pred_kdt_dt[pidx]['nodes_num']
#       pred_batch_num: int = self.pred_kdt_dt[pidx]['batch_points_num'] # multi-circle model
#       # print("prediction details", self.pred_kdt_dt[pidx]['batch_points_num'], self.pred_kdt_dt[pidx]['nodes_num'])

#       query_dist = binfo['radius'] + pinfo['radius'] + safe_clearance
#       result = btree.query_ball_tree(ptree, query_dist)
#       # print("len result list:", len(result)) # pairs of interactions.
#       for bp_idx, pp_idx_list in enumerate(result):
#         if len(pp_idx_list) == 0:
#           continue # skip when not interacted
#         #:param bp_idx: point index in behavior kdtree
#         #:param pp_idx_list: list of point index in prediction kdtree
#         bpoint = bpoints[bp_idx, :]

#         is_valid, info_location = self.xy_space.get_index(bpoint[:2])
#         agent_info_xy_locs.append([bpoint[0], bpoint[1]])

#         if not is_valid:
#           print("Warning, {} is out of the defined xy_space range.".format(bpoint[:2]))
#           continue # skip when the index is out of range

#         cache_info = []
#         for pp_idx in pp_idx_list:
#           #:param pp_idx: point index in prediction kdtree
#           ppoint = ppoints[pp_idx, :]
#           d0 = pp_idx % pred_batch_num # one-circle points amount
#           pred_traj_idx = math.floor(float(d0) / pred_traj_len)
#           pred_node_idx = d0 % pred_traj_len

#           # skip when having same (behavior path point idx, prediction_idx, prediciton_traj_idx)
#           repeat_key = (bp_idx, pidx, pred_traj_idx)
#           if repeat_key in repeat_checker_dict:
#             continue # skip repeat traj
#           repeat_checker_dict[repeat_key] = True

#           #:param agent_pose: pose of the interacted agent
#           agent_pose = ppoint
#           #:param tsvalue: s value of the interaction area
#           tsvalue = self.pred_records[pidx][pred_traj_idx]['sum_s_list'][pred_node_idx]
#           #:param tstamp: time stamp (seconds) of the interacted agent
#           tstamp = self.pred_records[pidx][pred_traj_idx]['tstamps'][pred_node_idx]

#           #:param tspeed: speed value of the interacted agent at the interaction area
#           tspeed0 = self.pred_records[pidx][pred_traj_idx]['speeds'][0]
#           tspeed = self.pred_records[pidx][pred_traj_idx]['speeds'][pred_node_idx]

#           cache_info.append(
#             formatter.get_list_of_data(
#               path_pose=bpoint,
#               pred_idx=pidx,
#               pred_initial_v=tspeed0,
#               pred_traj_idx=pred_traj_idx,
#               pred_traj_prob=probs[pred_traj_idx],
#               pred_traj_point_s=tsvalue,
#               pred_traj_point_t=tstamp,
#               pred_traj_point_yaw=agent_pose[2],
#               pred_traj_point_v=tspeed)
#             )

#         # get_data, array-like data for a prediction trajectory of an agent
#         get_data = np.array(cache_info, dtype=float)

#         if not info_location in self.agent_info_pool.keys():
#           self.agent_info_pool[info_location] = np.array([]).reshape(0, formatter.data_num())
#         self.agent_info_pool[info_location] = np.concatenate(
#           (self.agent_info_pool[info_location], get_data), axis=0)

#   if len(agent_info_xy_locs) > 0:
#     self.agent_info_xy_locs = np.array(agent_info_xy_locs)
#   else:
#     self.agent_info_xy_locs = np.array([]).reshape(0, 2)

# # Process behavioral kdtrees
# ego_width = ego_agent.info['width']
# ego_half_width = ego_width * 0.5
# ego_length = ego_agent.info['length']
# ego_half_length = ego_length * 0.5

# # ego: multi-circle model
# ego_circle_r = 1.0 * ego_half_width
# ego_front_x = ego_half_length - ego_half_width
# ego_back_x = -ego_half_length + ego_half_width    
# ego_offset_x_list = [ego_back_x, 0.0, ego_front_x]

# behavior_idx = -1 # all beahvior in one index
# # print("behavior[{}] visited xyyaws shape={}.".format(
# #   behavior_idx, behavior_visited_xyyaws.shape))
# if behavior_visited_xyyaws.shape[0] > 0:
#   self.ispace.add2kdtree('plan', behavior_idx, [behavior_visited_xyyaws], 
#     offset_x_list=ego_offset_x_list, radius=ego_circle_r)

# def get_behavior_visited_xyyaws(self, behaviors: Dict) -> np.ndarray:
#   '''
#   Discretize the visited space of the planned behaivor paths
#   :param behaviors: dict of behaviors obtained from behavior plan
#   :return: return all visited xyyaws of behaviors in numpy.ndarray with shape=(N, 3)
#   '''
#   behavior_idx: int = -1 # all behaviors are packed together
#   visited_xyyaws = np.array([]).reshape(0, 3) # initialize xyyaws
#   for behavior_key, dt in behaviors.items():
#     # print(behavior_key, dt['valid'], dt['lane_seq'], len(dt['candidate_paths']))
#     # dt: dict_keys( ['valid', 'lane_seq', 
#     #                 'is_truncated', 'frenet', 
#     #                 'from_to_sum_s', 'start_tpoint', 'start_fpoint', 
#     #                 'candidate_paths'])
#     if dt['valid'] == False:
#       continue

#     for _, path_dict in enumerate(dt['candidate_paths']):
#       visited_xyyaws = np.concatenate(
#         (visited_xyyaws, path_dict['path_samples_xyyawcurs'][:, :3]), axis=0)

#   self.xyyaw_discretize.reinit()
#   self.xyyaw_discretize.record_informations(
#     inputs=visited_xyyaws, info_idx=behavior_idx)

#   return self.xyyaw_discretize.get_grids_with_info(output_indexs=False)

# # Assign variables kdt_dt, is_pred according to mode
# kdt_dt: Dict[str, Any]= {}
# is_pred = False
# if mode == 'pred':
#   self.pred_kdt_dt[agent_idx] = {}
#   kdt_dt = self.pred_kdt_dt[agent_idx]
#   is_pred = True
# elif mode == 'plan':
#   self.plan_kdt_dt[agent_idx] = {}
#   kdt_dt = self.plan_kdt_dt[agent_idx]
# else:
#   raise ValueError("Illegal mode, not inside ['pred', 'plan']")

# # one trajectory point will be expaned to how many circles? (multi-circle model)
# num_expansion: int = len(offset_x_list)

# # >Fill information
# kdt_dt['info'] = {
#   'idx': agent_idx, 
#   'radius': radius, 
#   'expansion': num_expansion, 
#   'offset_x_list': offset_x_list, 
# }
# # >Initialize probability list
# kdt_dt['prob'] = [] # probability of each trajectory
# kdt_dt['nodes_num'] = None # traj node num of each trajectory

# # >Initialize points array
# kdt_dt['points'] = np.array([]).reshape(0, 3)

# for pred in predictions:
#   if isinstance(pred, np.ndarray): 
#     # mode == 'plan'
#     kdt_dt['prob'].append(1.0)
#     kdt_dt['points'] = np.concatenate((kdt_dt['points'], pred), axis=0)
#   elif 'prob' in pred.keys():
#     # mode == 'pred'
#     pred_points = pred['trajectory'] # original (13, 3)
#     tstamps = np.array([i * self.prediction_dt for i, _ in enumerate(pred_points)])

#     # update predition records
#     if not agent_idx in self.pred_records:
#       self.pred_records[agent_idx] = []
#     dxys = pred_points[1:, :2] - pred_points[:-1, :2]
#     piece_dists = np.linalg.norm(dxys, axis=1)
#     sum_s_list = np.zeros([1])
#     sum_s_list = np.hstack((sum_s_list, piece_dists))
#     for i in range(1, len(sum_s_list)):
#       sum_s_list[i] = sum_s_list[i-1] + sum_s_list[i]
#     speeds = (sum_s_list[1:] - sum_s_list[:-1]) / self.prediction_dt
#     speeds = np.hstack((speeds, np.array([speeds[-1]])))

#     self.pred_records[agent_idx].append(
#       { "points": pred_points,
#         'sum_s_list':  sum_s_list,
#         'tstamps': tstamps,
#         'speeds': speeds,
#       }
#     )

#     # update kdtree relevants
#     kdt_dt['prob'].append(pred['prob'])

#     if kdt_dt['nodes_num'] == None:
#       kdt_dt['nodes_num'] = pred_points.shape[0]

#     assert kdt_dt['nodes_num'] == pred_points.shape[0],\
#       "Fatal error, the prediction horizon is variational."
#     kdt_dt['points'] = np.concatenate(
#       (kdt_dt['points'], pred_points), axis=0)
#     # print("pred traj shape", pred_points.shape), (N, 3)
#   else:
#     raise ValueError("Unexpected type of predictions: {}.".format(type(pred)))

# # >Fill batch_points_num: the number of trajectory points before extend the
# #   trajectory points to multi-circle models
# kdt_dt['batch_points_num'] = kdt_dt['points'].shape[0]

# # batch apply transforms: apply multi-circle model
# if num_expansion >= 1:
#   sin_mat_unit = np.sin(kdt_dt['points'][:, 2])
#   cos_mat_unit = np.cos(kdt_dt['points'][:, 2])

#   # x = dx * self._cos_yaw + self._x
#   # y = dx * self._sin_yaw + self._y
#   add_x_mat = np.array([])
#   add_y_mat = np.array([])
#   for offset_x in offset_x_list:
#     add_x_mat = np.concatenate((add_x_mat, offset_x * cos_mat_unit))
#     add_y_mat = np.concatenate((add_y_mat, offset_x * sin_mat_unit))

#   kdt_dt['points'] = np.tile(kdt_dt['points'], (num_expansion, 1))
#   kdt_dt['points'][:, 0] = kdt_dt['points'][:, 0] + add_x_mat
#   kdt_dt['points'][:, 1] = kdt_dt['points'][:, 1] + add_y_mat

# # build kd tree with xy values
# if is_pred:
#   self.pred_kdtree[agent_idx] = KDTree(kdt_dt['points'][:, :2])
# else:
#   self.plan_kdtree[agent_idx] = KDTree(kdt_dt['points'][:, :2])
