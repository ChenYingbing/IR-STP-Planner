import copy
from typing import List
import numpy as np
import math

from utils.transform import XYYawTransform

TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE

def estimate_trajectory_yaws(input_traj: List):
  '''
  Estimate trajectory yaw angles according to their x, y values
  :parma input_traj: [[info], [x, y, yaw]_i, ...] shape = [len_traj, 3]
  '''
  output_traj = copy.deepcopy(input_traj)
  len_traj = output_traj.shape[0]
  for ti in range(1, len_traj):
    ti_n1 = ti - 1
    ti_p1 = ti + 1
    if ti_p1 < len_traj:
      tp0 = output_traj[ti, :][0:2]
      tp1 = output_traj[ti_p1, :][0:2]

      dx = tp1[0] - tp0[0]
      dy = tp1[1] - tp0[1]
      output_traj[ti, 2] = math.atan2(dy, dx)
    elif ti_n1 >= 1: # first frame is trajectory info
      tp0 = output_traj[ti_n1, :][0:2]
      tp1 = output_traj[ti, :][0:2]

      dx = tp1[0] - tp0[0]
      dy = tp1[1] - tp0[1]
      output_traj[ti, 2] = math.atan2(dy, dx)

  return output_traj

def normalize_trajectory(input_traj: np.ndarray):
  '''
  Normalize trajectory given array like trajectory
  @note where input_traj do not contains trajectory info
  '''
  if input_traj.shape[0] >= 1:
    state0 = input_traj[0, :]

    xyyaw0 = XYYawTransform(x=state0[0], y=state0[1], yaw_radian=state0[2])
    inv_xyyaw0 = copy.copy(xyyaw0)
    inv_xyyaw0.inverse()

    get_traj = []
    for state in input_traj:
      xyyaw = XYYawTransform(
        x=state[0], 
        y=state[1], 
        yaw_radian=state[2])
      start2xyyaw = inv_xyyaw0.multiply_from_right(xyyaw)
      get_traj.append([start2xyyaw._x, start2xyyaw._y, start2xyyaw._yaw])
    input_traj = np.array(get_traj)

  return input_traj