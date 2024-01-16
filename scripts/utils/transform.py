#!/usr/bin/env python

import numpy as np
import math

M_PI_2 = 1.5707963267948966
M_PI = 3.141592653589793238462643383279502884
M_2PI = M_PI * 2.0

def get_rotation(mat):
  to_degree = 57.29577951471995

  pitch = math.asin(mat[2][0]) * to_degree
  roll = -math.atan2(mat[2][1], mat[2][2]) * to_degree
  yaw = math.atan2(mat[1][0], mat[0][0]) * to_degree
  return [pitch, roll, yaw]

# @note, input yaw in init() is in degree, _yaw is in radian
class XYYawTransform():
  def __init__(self, x=0.0, y=0.0, yaw_degree=0.0, yaw_radian=None):
    if yaw_radian:
        self.set_values(x, y, yaw_radian * 57.29577951471995)
    else:
        self.set_values(x, y, yaw_degree)

  def set_values(self, x, y, yaw_degree):
    self._x = x
    self._y = y

    yaw = yaw_degree / 57.29577951471995 # yaw input is in degrees
    self._yaw = yaw
    self._sin_yaw = math.sin(yaw)
    self._cos_yaw = math.cos(yaw)

  def set_from_mat(self, mat):
    self.set_values(mat[0][3],
                    mat[1][3], get_rotation(mat)[2])

  def get_mat(self):
    mat = np.identity(4)
    sin_theta = math.sin(self._yaw)
    cos_theta = math.cos(self._yaw)

    mat[0][0] = cos_theta
    mat[0][1] = -sin_theta
    mat[1][0] = sin_theta
    mat[1][1] = cos_theta
    mat[0][3] = self._x
    mat[1][3] = self._y
    return mat

  def get_list(self):
    return [self._x, self._y, self._yaw]

  def distance2(self, xyyaw) -> float:
    '''
    return cartesian distance to another xyyaw point
    '''
    dx = xyyaw._x - self._x
    dy = xyyaw._y - self._y
    
    return math.sqrt(dx**2 + dy**2)

  # multiplication from right, mat[self] * mat[right]
  def multiply_from_right(self, r_xyyaw):
    xyyaw = XYYawTransform()

    xyyaw._x = r_xyyaw._x * self._cos_yaw - r_xyyaw._y * self._sin_yaw + self._x
    xyyaw._y = r_xyyaw._x * self._sin_yaw + r_xyyaw._y * self._cos_yaw + self._y

    yaw = self._yaw + r_xyyaw._yaw
    if (yaw > M_PI):
        yaw = yaw - M_2PI
    elif (yaw < -M_PI):
        yaw = yaw + M_2PI
    xyyaw._yaw = yaw

    xyyaw._sin_yaw = math.sin(yaw)
    xyyaw._cos_yaw = math.cos(yaw)
    return xyyaw

  def fast_multiply_from_right(self, relative_x: float, relative_y: float):
    xyyaw = XYYawTransform()

    xyyaw._x = relative_x * self._cos_yaw - relative_y * self._sin_yaw + self._x
    xyyaw._y = relative_x * self._sin_yaw + relative_y * self._cos_yaw + self._y
    xyyaw._yaw = self._yaw

    return xyyaw

  def inverse(self):
    inv_mat = np.linalg.inv(self.get_mat())
    self.set_from_mat(inv_mat)

  # map from global frame to this frame, vx, vy
  def get_this_frame_velocities(self, g_vx, g_vy):
    l_vx = self._cos_yaw * g_vx + self._sin_yaw * g_vy
    l_vy = -self._sin_yaw * g_vx + self._cos_yaw * g_vy
    return l_vx, l_vy
