#!/usr/bin/env python

from math import fmod
import math
import numpy as np

_M_PI = 3.14159265358979323846
_double_M_PI = 6.283185307179586

def get_normalized_angle(angle: float) -> float:  
  a = fmod(angle + _M_PI, _double_M_PI)
  if (a < 0.0):
    a = a + _double_M_PI
  return (a - _M_PI)

def get_normalized_angles(angle_array: np.ndarray) -> np.ndarray:  
  a_array = np.fmod(angle_array + _M_PI, _double_M_PI)
  for i, a in enumerate(a_array):
    if (a < 0.0):
      a_array[i] = a + _double_M_PI
  return (a_array - _M_PI)

def get_xyyaw_distance(xyyaw0, xyyaw1, coef: float = 1.0) -> float:
  dxyyaw = np.array(xyyaw0) - np.array(xyyaw1)
  if dxyyaw.ndim == 1:
    a = fmod(dxyyaw[2] + _M_PI, _double_M_PI)
    if (a < 0.0):
      a = a + _double_M_PI
    dxyyaw[2] = a - _M_PI
    distance = math.sqrt(dxyyaw[0]*dxyyaw[0] + dxyyaw[1]*dxyyaw[1] + dxyyaw[2]*dxyyaw[2])
  else:
    dxyyaw[:, 2] = get_normalized_angle(dxyyaw[:, 2]) * coef
    distance = np.linalg.norm(dxyyaw, axis=1).min()
  return distance

def get_mean_yaw_value(yaws: np.ndarray) -> float:
  yaw0 = yaws[0]
  dyaws = get_normalized_angles(yaws - yaw0)
  mean_dyaw = np.mean(dyaws)

  mean_yaw = get_normalized_angle(yaw0 + mean_dyaw)
  return mean_yaw
