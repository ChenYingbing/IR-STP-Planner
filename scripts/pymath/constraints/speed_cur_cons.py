import math
import numpy as np

TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE

class SpeedCurvatureCons:
  @staticmethod
  def get_steer_speed_limit(wheel_base: float, steer: float, u: float) -> float:
    '''
    :param wheel_base: wheel base of vehicle
    :param steer: steer of vehicle in radian
    :param u: friction coefficient
    return speed limit (m/s) given steer and u
    '''
    # @brief following equations below
    #    u * m * g = F = m * v^2 / R
    #    u * g = F =v^2 / R
    #    R = L / steer
    # >> u * g = F =v^2 * tan(steer) / L
    g = 9.8
    abs_steer = math.fabs(steer)

    if abs_steer < 1e-5:
      return 1e+3
    else:
      return math.sqrt(u * g * (wheel_base / math.tan(abs_steer)))

  @staticmethod
  def get_curvature_speed_limit(curvature: float, u: float) -> float:
    '''
    :param curvature: the curvature input
    :param u: friction coefficient
    return speed limit (m/s) given curvature and u
    '''
    g = 9.8
    abs_curvature = math.fabs(curvature)
    if abs_curvature < 1e-5:
      return 1e+3
    else:
      return math.sqrt(u * g / abs_curvature)

  @staticmethod
  def get_curvature_speed_limits(curvatures: np.ndarray, u: float, min_speed_limit: float) -> np.ndarray:
    '''
    :param curvatures: array of curvatures
    :param u: friction coefficient
    :param min_speed_limit: minimum speed limit being set
    return speed limits (m/s, np.ndarray) given curvatures and u
    '''
    g = 9.8
    abs_curvatures = np.fabs(curvatures)
    speed_limits = np.ones_like(curvatures) * 1e+3

    update_locs = (speed_limits >= 1e-5)
    cache = np.sqrt(np.ones_like(u * g) / abs_curvatures)
    speed_limits[update_locs] = cache[update_locs]

    speed_limits[speed_limits < min_speed_limit] = min_speed_limit
    return speed_limits

  @staticmethod
  def get_curvature_limit(max_speed: float, u: float, speed_threshold: float = 1.0) -> float:
    '''
    :param u: friction coefficient
    :param max_speed: maximum speed (m/s)
    :param speed_threshold: minimum value of max_speed
    return absolute curvature limit
    '''
    g = 9.8
    abs_cur_max = (u * g) / (max(max_speed, speed_threshold) **2)
    return abs_cur_max
