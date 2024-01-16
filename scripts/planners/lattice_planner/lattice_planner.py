from turtle import back
from typing import Tuple
from pymath.curves.quintic import QuinticPolynomial
from type_utils.frenet_path import FrenetPath
from pymath.curves.bspline import SplineCurve
import math
import numpy as np

class LatticePlanner:
  def __init__(self, curvature_limit: float = 0.3333,
                     cur_cost_weight: float = 1.25,
                     merge_t_cost_weight: float = 1.0,
                     d_offset_cost_weight: float = 0.25):
    '''
    Generate paths along a frenet path using lattice
    :param cur_cost_weight: curvature cost
    :param merge_t_cost_weight: time cost when merging
    '''
    self.curvature_limit = curvature_limit
    self.cost_weights = {
      "cur_cost_weight": cur_cost_weight, 
      "merge_t_cost_weight": merge_t_cost_weight, 
      "d_offset_cost_weight": d_offset_cost_weight
    }

  def gen_ds_paths(self, fpath: FrenetPath,
                         start_s_pv: Tuple[float, float],
                         start_d_pva: Tuple[float, float, float],
                         s_range: Tuple[float, float], 
                         lon_s_reso: float, 
                         d_range: Tuple[float, float],
                         lat_d_reso: float,
                         lon_max_sample_num: int = 1,
                         ref_sample_dd_interval: float = 0.1,
                         sample_interval_s: float = 1.0):
    '''
    Generate batch of d-s paths along fpath
    :param fpath: given frenet path
    :param start_s_pv: start s, position, velocity
    :param start_d_pva: start d, position, velocity, acc of d.
    :param s_range: [s_from, s_to] longitudinal sample range
    :param lon_s_reso: longitudinal sample resolution
    :param d_range: [d_from, d_to] lateral sample range
    :param lat_d_reso: lateral sample resolution
    :param ref_sample_dd_interval: reference lateral speed (dd/ds) for sampling
    '''
    s_range[1] = min(fpath.max_sum_s(), s_range[1])

    s_sample_dist = s_range[1] - s_range[0]
    d_sample_dist = d_range[1] - d_range[0]
    assert s_sample_dist >= 0.0, "Input Error: s_sample_dist < 0.0"
    assert d_sample_dist >= 0.0, "Input Error: d_sample_dist < 0.0"

    s_sample_num :int= math.ceil(s_sample_dist / lon_s_reso) + 1
    end_s_list = np.linspace(s_range[0], s_range[1], s_sample_num)
    end_d_list = np.arange(start=d_range[0], stop=d_range[1]+1e-6, step=lat_d_reso)

    back_s = s_range[-1]
    get_paths = []

    print("Gen_ds_paths::sample_lateral_num={}; sample_lon_num={}.".format(
      len(end_d_list), len(end_s_list)))
    for offset_d in end_d_list:  # merging d offset
      d_move_dist = math.fabs(offset_d - start_d_pva[0])
      lat_d_paths = []
      cond2sample: int = 1e+3

      # s-d: 5-order poly -> 100; s-t: 4-order -> 100;
      # 100 * 100 > 10000 >>> cartesian: remove invalid / select best.
      for mer_s in end_s_list[1:]: # mer_s: merging point (from near to far)
        # extract d-s polyfit part
        s_move_dist = mer_s - start_s_pv[0]
        ref_dd = d_move_dist / s_move_dist
        
        # [cond1]: check sample conditions
        cond_value: int = math.floor(ref_dd / ref_sample_dd_interval)
        if cond_value < cond2sample:
          cond2sample = cond_value # each interval enables sample one time
        else:
          # print("skip sample with s={}, d={}, and red_dd={}.".format(
          #   s_move_dist, d_move_dist, ref_dd))
          continue # skip sampling
        # [cond2]: exceed merge_time horizon
        expect_lon_merge_T = math.fabs((mer_s - start_s_pv[0]) / max(start_s_pv[1], 3.0)) # 3.0 is the min_v

        expect_lat_merge_T = math.fabs(start_d_pva[0]) / max(1.0, math.fabs(start_d_pva[1])) # 1.0 is the ref merge v
        merge_t_cost = math.fabs(expect_lon_merge_T - expect_lat_merge_T)

        poly_s_max = mer_s - start_s_pv[0]
        poly5d = QuinticPolynomial(start_d_pva, (offset_d, 0.0, 0.0), T=poly_s_max)
        sd_path = poly5d.extract_path(T1=poly_s_max, dt=0.5)[:, :2] # (s, d) values
        s_path = sd_path[:, 0] + start_s_pv[0]
        d_path = sd_path[:, 1]

        # append no-polyfit part
        extra_s_reso = 0.5
        extra_s_dist = (back_s - mer_s)
        if extra_s_dist >= extra_s_reso:
          extra_num = math.ceil(extra_s_dist / extra_s_reso) + 1
          extra_s_path = np.linspace(mer_s, back_s, extra_num)[1:] # remove first point
          s_path = np.concatenate((s_path, extra_s_path))
          d_path = np.concatenate((d_path, np.ones_like(extra_s_path)*offset_d))

        sd_path = np.transpose(np.vstack((s_path, d_path)))
        path_xy_points = fpath.get_cartesian_points(sd_path)[:, :2]
        poly_path = SplineCurve(path_xy_points)

        ref_sample_num :int= math.ceil(
          poly_path.get_max_sum_s() / sample_interval_s) + 1

        # merge more fast, cost more slow
        d_offset_cost = math.fabs(offset_d - start_d_pva[0])
        extra_cost = self.cost_weights['merge_t_cost_weight'] * merge_t_cost +\
                      self.cost_weights['d_offset_cost_weight'] * d_offset_cost

        s_sample_interval = 1.0
        is_legal, get_cost, s_samples, xyyawcurs_samples = \
          poly_path.evaluate_and_sample(
            ref_sample_num, self.curvature_limit, start_s_pv[1],
            extra_cost, [self.cost_weights['cur_cost_weight'], s_sample_interval])

        # print("poly_path", poly_path.get_max_sum_s(), is_legal)
        # print("start_s={:.2f}, end_sd={:.2f}/{:.1f}".format(
        #   start_s, mer_s, end_d), sd_path.shape)
        if is_legal:
          # print("lattice sample from d=[{:.1f}, {:.1f}], s_v0={:.1f}; to "
          #       "s={:.1f}, d={:.1f}, dd={:.1f}, and cost={:.3f}.".format(
          #       start_d_pva[0], start_d_pva[1], start_s_pv[1], 
          #       s_move_dist, d_move_dist, ref_dd, get_cost))
          lat_d_paths.append({
            'frenet_from_s': start_s_pv[0],
            'frenet_to_s': back_s,
            'frenet_merge_s': mer_s,
            'offset_d': offset_d,
            'cost_path': get_cost,
            'poly_path': poly_path,
            'path_samples_s': s_samples,
            'path_samples_s_interval': s_sample_interval,
            'path_samples_xyyawcurs': xyyawcurs_samples,
          })

      lat_d_paths = sorted(lat_d_paths, key=lambda p: p['cost_path'])

      # append from cost min to cost max, with number < lon_max_sample_num limit
      for d_path in lat_d_paths:
        if len(get_paths) < lon_max_sample_num:
          get_paths.append(d_path)
        else:
          break

    get_paths = sorted(get_paths, key=lambda p: p['cost_path']) 
    print("Gen_ds_paths::result path amount=", len(get_paths))
    return get_paths
    