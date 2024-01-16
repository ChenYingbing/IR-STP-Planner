from operator import index
from random import sample
from typing import Tuple
import numpy as np
import scipy.interpolate as scipy_interpolate
import matplotlib.pyplot as plt
import math
import copy

from pymath.constraints.speed_cur_cons import SpeedCurvatureCons
from pymath.curves.cubic_spline import CubicSpline

class SplineCurve:
  def __init__(self, xy_points: np.ndarray,
                     knots_distance: float = 8.0) -> None:
    '''
    Piecewise spline curve
    :param xy_points: input x, y points with shape=(num, 2)
    :param splrep_s: value is used for splrep(s=splrep_s)
    '''
    # points_s = np.linspace(0, 10.0, 200)
    # x_rand = (np.random.rand(200) - 0.5) * 2.0
    # y_rand = (np.random.rand(200) - 0.5) * 2.0
    # x = 5.0*points_s + 1.0
    # y = 1.0*points_s + 0.5
    # x[10:-10] += x_rand[10:-10]
    # y[10:-10] += y_rand[10:-10]
    # test_xy = np.transpose(np.array([x, y]))
    # xy_points = test_xy
    xy_points = copy.copy(xy_points)
    self.ori_xy_points = copy.copy(xy_points[0, :])

    xy_points -= self.ori_xy_points
    euler_dists = np.linalg.norm((xy_points[1:] - xy_points[0:-1]), axis=1)
    sum_s_list = np.zeros([1])
    sum_s_list = np.hstack((sum_s_list, euler_dists))
    for i in range(1, len(sum_s_list)):
      sum_s_list[i] = sum_s_list[i-1] + sum_s_list[i]

    self.sum_s_list = sum_s_list
    self.max_sum_s = sum_s_list[-1]

    # Extracct t_array, x_array, y_array
    t_array, x_array, y_array = [], [], []
    threshold_s = -1e-6
    for idx, get_s in enumerate(sum_s_list.tolist()):
      if get_s >= threshold_s:
        t_array.append(get_s)
        x_array.append(xy_points[idx, 0])
        y_array.append(xy_points[idx, 1])
        threshold_s += knots_distance
    if (len(t_array) > 2) and ((self.max_sum_s - t_array[-1]) <= (knots_distance * 0.5)):
      t_array.pop()
      x_array.pop()
      y_array.pop()
    t_array.append(sum_s_list[-1])
    x_array.append(xy_points[-1, 0])
    y_array.append(xy_points[-1, 1])

    self.x_spl = CubicSpline(t_array, x_array)
    self.y_spl = CubicSpline(t_array, y_array)

    # self.x_spl = scipy_interpolate.splrep(
    #   sum_s_list, xy_points[:, 0], s=splrep_s, # task=-1, 
    #   t=t_array[1:-1], k=polyfit_k)
    # self.y_spl = scipy_interpolate.splrep(
    #   sum_s_list, xy_points[:, 1], s=splrep_s, # task=-1, 
    #   t=t_array[1:-1], k=polyfit_k)

    self.poly_xys = np.transpose(np.array([x_array, y_array]))
    self.poly_xys += self.ori_xy_points

  def get_max_sum_s(self) -> float:
    return self.max_sum_s

  def get_sum_s_list(self) -> np.ndarray:
    return self.sum_s_list

  def get_sample_xy(self, sample_s: np.ndarray, order: int=0) -> np.ndarray:
    '''
    Return x, y values at certain order
    '''
    sample_s = np.clip(sample_s, 0.0, self.max_sum_s -1e-6)

    assert order == 0, "Unsupport order > 0 situation"
    sample_idxs = [self.x_spl.search_index(ss) for ss in sample_s]

    get_x = self.x_spl.my_calc(sample_s, sample_idxs)
    get_y = self.y_spl.my_calc(sample_s, sample_idxs)

    # print("\n*********************")
    # print("sample_s", sample_s.shape, get_x.shape, get_y.shape)
    get_xy = np.vstack((get_x, get_y))
    get_xy = np.transpose(get_xy)

    # print("get_sample_xy", get_xy.shape, self.ori_xy_points.shape)
    if (order == 0):
      get_xy[:, 0:2] += self.ori_xy_points

    return get_xy

  def get_sample_xyyaw(self, sample_s: np.ndarray) -> np.ndarray:
    '''
    Return xxyaw array (num, 3=[x, y, yaw])
    '''
    sample_s = np.clip(sample_s, 0.0, self.max_sum_s -1e-6)

    sample_idxs = [self.x_spl.search_index(ss) for ss in sample_s]

    x = self.x_spl.my_calc(sample_s, sample_idxs)
    y = self.y_spl.my_calc(sample_s, sample_idxs)
    
    dx = self.x_spl.my_calcd(sample_s, sample_idxs)
    dy = self.y_spl.my_calcd(sample_s, sample_idxs)
    yaw = np.arctan2(dy, dx)

    rarray = np.vstack((x, y, yaw))
    rarray = np.transpose(rarray)
    rarray[:, 0:2] += self.ori_xy_points
    return rarray

  def get_sample_xyyawcur(self, sample_s: np.ndarray) -> np.ndarray:
    '''
    Return xxyaw and curvature array (num, 4=[x, y, yaw, curvature])
    '''
    sample_s = np.clip(sample_s, 0.0, self.max_sum_s -1e-6)

    sample_idxs = [self.x_spl.search_index(ss) for ss in sample_s]

    x = self.x_spl.my_calc(sample_s, sample_idxs)
    y = self.y_spl.my_calc(sample_s, sample_idxs)
    dx = self.x_spl.my_calcd(sample_s, sample_idxs)
    dy = self.y_spl.my_calcd(sample_s, sample_idxs)
    ddx = self.x_spl.my_calcdd(sample_s, sample_idxs)
    ddy = self.y_spl.my_calcdd(sample_s, sample_idxs)

    # print("\n >>>>>>>>>>>>>>>>>>>>>>>.")
    # print(sample_s.shape, x.shape, dx.shape)
  
    yaw = np.arctan2(dy, dx)
    curvature = (dx* ddx - dy* ddy) / np.power(dx** 2 + dy** 2, 3 / 2)

    rarray = np.vstack((x, y, yaw, curvature))
    rarray = np.transpose(rarray)
    rarray[:, 0:2] += self.ori_xy_points
    return rarray

  def get_curvature(self, sample_s: np.ndarray) -> np.ndarray:
    '''
    Return curvature values
    '''
    sample_s = np.clip(sample_s, 0.0, self.max_sum_s -1e-6)

    sample_idxs = [self.x_spl.search_index(ss) for ss in sample_s]

    dx = self.x_spl.my_calcd(sample_s, sample_idxs)
    dy = self.y_spl.my_calcd(sample_s, sample_idxs)
    ddx = self.x_spl.my_calcdd(sample_s, sample_idxs)
    ddy = self.y_spl.my_calcdd(sample_s, sample_idxs)

    # dx = scipy_interpolate.splev(sample_s, self.x_spl, der=1)
    # dy = scipy_interpolate.splev(sample_s, self.y_spl, der=1)
    # ddx = scipy_interpolate.splev(sample_s, self.x_spl, der=2)
    # ddy = scipy_interpolate.splev(sample_s, self.y_spl, der=2)

    curvature = (dx* ddx - dy* ddy) / np.power(dx** 2 + dy** 2, 3 / 2)

    return curvature

  def evaluate_and_sample(self, 
      sample_num: int,
      curvature_limit: float,
      reference_v: float,
      extra_cost: float = 0.0,
      cost_weights: Tuple[float, float] = [1.0, 1.0]
    ) -> Tuple[bool, float, np.array, np.array]:
    '''
    Evaluate the spline at certain number of sample points
    :param sample_num: num of sample points to evaluate
    return flag, cost, np.array: where 
      1. flag: represents whether the spline is legal or not
      2. cost: is the evaluation cost
      3. np.array: are the sampled s_array, and xyyawcur points
    '''
    s_array = np.linspace(0.0, self.max_sum_s, sample_num)
    abs_curs = np.abs(self.get_curvature(s_array))

    ref_max_cur = SpeedCurvatureCons.get_curvature_limit(reference_v, u=0.1)
    # mean_init_abs_cur = np.mean(abs_curs[:5]) # previous 5 units
    # is_illegal = (np.sum((abs_curs >= curvature_limit) * 1.0) > 1e-6) or (mean_init_abs_cur > (ref_max_cur * 3.0))
    is_illegal = (np.sum((abs_curs >= curvature_limit) * 1.0) > 1e-6)
    cost_cur = max(np.max(abs_curs - ref_max_cur), 0.0)
    get_cost = cost_weights[0] * cost_cur + cost_weights[1] * extra_cost

    # if not is_illegal:
    #   print("get_cost={} with".format(get_cost), cost_cur, extra_cost)

    return (not is_illegal), get_cost, s_array, self.get_sample_xyyawcur(s_array)

  def test_plot(self) -> None:
    s_array = np.linspace(0.0, self.max_sum_s, 100)
    get_samples = self.get_sample_xyyawcur(s_array)

    fig = plt.figure()
    plt.subplot(131)
    plt.plot(self.poly_xys[:, 0], self.poly_xys[:, 1], 'b.')
    plt.plot(get_samples[:, 0], get_samples[:, 1], 'r-')
    plt.subplot(132)
    plt.plot(s_array, get_samples[:, 3], 'b-')
    plt.subplot(133)
    plt.plot(s_array, get_samples[:, 2], 'g-')
    plt.show()

  # def my_example_func():
  #   sum_lon_dist = 10.0
  #   point_num = 10

  #   points_s = np.linspace(0, sum_lon_dist, point_num)
  #   sample_p= np.linspace(0.0, 1.0, 100)

  #   knots_num = 5

  #   x_rand = (np.random.rand(point_num) - 0.5) * 2.0
  #   y_rand = (np.random.rand(point_num) - 0.5) * 2.0
  #   x = 5.0*points_s + 1.0 + x_rand
  #   y = 2.0*points_s + 0.5 + y_rand

  #   tck, u = splprep(np.vstack((x,y)), s=1e-3, k=3, nest=knots_num)

  #   px, py = splev(sample_p, tck)
  #   dx, dy = splev(sample_p, tck, der=1)
  #   ddx, ddy = splev(sample_p, tck, der=2)
  #   curvature = np.abs(dx* ddx - dy* ddy) / np.power(dx** 2 + dy** 2, 3 / 2)

  #   print("len x, dx, ddx, cur", len(px), len(dx), len(ddx), len(curvature))
