from typing import Tuple, Dict

from pydantic import TupleError
import numpy as np
import math
import copy

from utils.angle_operation import get_normalized_angle
from type_utils.trajectory_point import TrajectoryPoint
from pymath.curves.bspline import SplineCurve
from utils.transform import XYYawTransform

class FrenetPath:
  '''
  Class to describe the frenet path
  '''
  def __init__(self, path_xyyaws: np.ndarray, left_xyyaws: np.ndarray, right_xyyaws: np.ndarray) -> None:
    '''
    Init given a list of xy points
    :param path_xyyaws: shape=(num_path_node, 3:[x, y, yaw])
    '''
    assert path_xyyaws.shape[0] == left_xyyaws.shape[0], "frenet_path left-size unmatched = {}".format(
      [path_xyyaws.shape, left_xyyaws.shape])
    assert path_xyyaws.shape[0] == right_xyyaws.shape[0], "frenet_path right-size unmatched = {}".format(
      [path_xyyaws.shape, right_xyyaws.shape])

    self.ori_left_xyyaws = left_xyyaws
    self.ori_mid_xyyaws = path_xyyaws
    self.ori_right_xyyaws = right_xyyaws

    # fill path_xys
    dist2left_edges = np.linalg.norm((left_xyyaws - path_xyyaws)[:, :2], axis=1)
    dist2right_edges = np.linalg.norm((right_xyyaws - path_xyyaws)[:, :2], axis=1)
    path_xys = path_xyyaws[:, 0:2]

    self.spline = SplineCurve(path_xys)       # continuous values
    self.path_xys = path_xys                  # discrete values
    self.path_left_bounds = dist2left_edges
    self.path_right_bounds = dist2right_edges

    # fill frenet_path: format follows 
    #   [[s, x, y, yaw, sin(yaw), cos(yaw)], ...]
    sum_s_list = self.spline.get_sum_s_list()
    frenet_path = [[s, xyyaw[0], xyyaw[1], xyyaw[2], math.sin(xyyaw[2]), math.cos(xyyaw[2])] \
                    for s, xyyaw in zip(sum_s_list, path_xyyaws)]
    self.frenet_path = np.array(frenet_path)

    # self.spline.test_plot()

  def max_sum_s(self) -> float:
    '''
    Return maximum longitudinal accumulated distance.
    '''
    if self.frenet_path.shape[0] == 0:
      return 0.0
    else:
      max_lon_s = self.frenet_path[-1, 0]
      return max_lon_s

  def sum_s(self, xy: Tuple[float, float], epsilon: float = 0.1) -> float:
    '''
    Return accumulated distance at xy (using bisection search)
    '''
    # get more accurate lb and ub
    s_list = np.arange(0.0, self.spline.max_sum_s+1e-3, 1.0)
    dists_list = np.linalg.norm(
      self.spline.get_sample_xy(s_list) - xy, axis=1)
    loc_min = np.argmin(dists_list)

    # find the min distance point using bisection method given lb, ub
    lb = 0.0
    ub = self.spline.max_sum_s
    if loc_min > 0:
      lb = s_list[loc_min-1]
    if loc_min < (s_list.shape[0] - 1):
      ub = s_list[loc_min+1]

    step = (ub - lb) / 2.0
    mid = lb + step
    xy = np.array(xy)

    # left/mid/right values
    lmr_value = np.linalg.norm(
      self.spline.get_sample_xy(np.array([lb, mid, ub])) - xy, axis=1)

    while (math.fabs(ub - lb) > epsilon):
      l_value = lmr_value[0]
      m_value = lmr_value[1]
      # r_value = lmr_value[2]

      cache_min = np.min(lmr_value)
      # print("lmu={:.1f},{:.1f},{:.1f}".format(lb, mid, ub), "v_step", cache_min, step)
      # print("lmr_value=", lmr_value)
      step *= 0.5
      if math.fabs(cache_min - l_value) < 1e-6:
        # closing to l_value
        mid = lb + step
        ub = copy.copy(mid)
        
        lmr_value = np.linalg.norm(
          self.spline.get_sample_xy(np.array([lb, mid, ub])) - xy, axis=1)
      elif math.fabs(cache_min - m_value) < 1e-6:
        # closing to m_value
        lb = mid - step * 1.5 # limit the shrink extent
        ub = mid + step * 1.5

        lmr_value = np.linalg.norm(
          self.spline.get_sample_xy(np.array([lb, mid, ub])) - xy, axis=1)
      else:
        # closing to r_value
        lb = copy.copy(mid)
        mid = ub - step

        lmr_value = np.linalg.norm(
          self.spline.get_sample_xy(np.array([lb, mid, ub])) - xy, axis=1)

    return (lb + ub) / 2

  def remained_sum_s(self, xy: Tuple[float, float]) -> float:
    '''
    Return distance to max_sum_s()
    '''
    return (self.max_sum_s() - self.sum_s(xy))

  def get_spline(self) -> SplineCurve:
    '''
    Return object of spline, which is used for further query
    '''
    return self.spline

  def get_refpoint(self, xy: Tuple[float, float], epsilon: float=0.1) -> Tuple:
    '''
    Return frenet reference point given trajectory point (cartesian frame).
    '''
    get_s = self.sum_s(xy, epsilon=epsilon)
    ref_xyyawcur = self.spline.get_sample_xyyawcur([get_s])
    
    return get_s, ref_xyyawcur[0, :]

  def get_cartesian_points(self, sd_values: np.ndarray) -> np.ndarray:
    '''
    Return cartesian (x, y) points given (s, d) values
    return shape = (num, 3), = [[x, y, yaw], ...]
    '''
    xyyawcur_array = self.spline.get_sample_xyyawcur(sd_values[:, 0])
    get_xyyaws = []
    for xyyawcur, d in zip(xyyawcur_array, sd_values[:, 1]):
      ref_sin = math.sin(xyyawcur[2])
      ref_cos = math.cos(xyyawcur[2])

      get_x = -d * ref_sin + xyyawcur[0]
      get_y = d * ref_cos + xyyawcur[1]
      get_xyyaws.append([get_x, get_y, xyyawcur[2]])

    # print("\n************************************")
    # print(np.array(get_xyyaws).shape)
    return np.array(get_xyyaws)

  def get_frenet_point(self, tpoint: TrajectoryPoint) -> Dict:
    '''
    Return frenet point given trajectory point (cartesian frame).
    '''
    tstate = tpoint.state()
    # extract tstate
    tstamp = tstate['timestamp']
    tx = tstate['pos_x']
    ty = tstate['pos_y']
    tyaw = tstate['pos_yaw']
    tsteer = tstate['steer']
    tv = tstate['velocity']
    tacc = tstate['acceleration']

    ref_s, rxyyawcur = self.get_refpoint((tx, ty))
    ref_x, ref_y, ref_yaw, ref_cur = rxyyawcur
    ref_sin = math.sin(ref_yaw)
    ref_cos = math.cos(ref_yaw)

    dtheta = get_normalized_angle(tyaw - ref_yaw)
    cos_dtheta = math.cos(dtheta)
    tan_dtheta = math.tan(dtheta)

    # 0-order
    dx = tx - ref_x
    dy = ty - ref_y
    cross_rd_nd = ref_cos * dy - ref_sin * dx
    abs_d = math.sqrt(dx*dx + dy*dy)
    d = abs_d
    if (cross_rd_nd < 0.0):
      d = -abs_d
    
    s = ref_s

    # 1-order
    one_minus_kappa_r_d = 1.0 - ref_cur * d
    
    ds = tv * cos_dtheta / one_minus_kappa_r_d
    dd = one_minus_kappa_r_d * tan_dtheta # dd / ds

    # 2-order
    ref_kappa = ref_cur
    ref_dkapaa = 0.0
    kappa = 0.0 # state kappa TODO(abing): transfer
    delta_theta_prime = one_minus_kappa_r_d / cos_dtheta * kappa - ref_kappa
    kappa_r_d_prime = ref_dkapaa * d + ref_kappa * dd

    dds = (tacc * cos_dtheta - ds * ds * (dd * delta_theta_prime - kappa_r_d_prime)) / one_minus_kappa_r_d
    ddd = -kappa_r_d_prime * tan_dtheta +\
          one_minus_kappa_r_d / cos_dtheta / cos_dtheta * (kappa * one_minus_kappa_r_d / cos_dtheta - ref_kappa)

    return {
      'ref_s': s,
      'ref_xyyawcur': rxyyawcur, # [x, y, yaw, cur]
      'frenet': [s, ds, dds, d, dd, ddd],
    }