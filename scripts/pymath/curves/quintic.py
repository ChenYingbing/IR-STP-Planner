from typing import Tuple
import numpy as np
import copy

class QuinticPolynomial:
  def __init__(self, sxva: Tuple[float, float, float], 
                     exva: Tuple[float, float, float], 
                     T: float):
    x0, v0, a0 = sxva
    x1, v1, a1 = exva

    A = np.array([[T ** 3, T ** 4, T ** 5],
                  [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                  [6 * T, 12 * T ** 2, 20 * T ** 3]])
    b = np.array([x1 - x0 - v0 * T - a0 * T ** 2 / 2,
                  v1 - v0 - a0 * T,
                  a1 - a0])
    X = np.linalg.solve(A, b)

    self.T = T
    self.a0 = x0
    self.a1 = v0
    self.a2 = a0 / 2.0
    self.a3 = X[0]
    self.a4 = X[1]
    self.a5 = X[2]

  def calc_xt(self, t: float):
    xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
            self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

    return xt

  def calc_dxt(self, t: float):
    dxt = self.a1 + 2 * self.a2 * t + \
        3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

    return dxt

  def calc_ddxt(self, t: float):
    ddxt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

    return ddxt

  def calc_dddxt(self, t: float):
    dddxt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

    return dddxt

  def extract_path(self, T1: float, dt: float) -> np.ndarray:
    '''
    Extract path given t_from > t_to
    @note when T1 > self.T, forcibly set x = calc_xt(self.T) value
    '''
    get_path = []
    last_px = self.calc_xt(self.T)
    for t in np.arange(0.0, T1 + 1e-6, dt):
      px, dx, ddx, dddx = 0.0, 0.0, 0.0, 0.0
      if t <= self.T:
        px = self.calc_xt(t)
        dx = self.calc_dxt(t)
        ddx = self.calc_ddxt(t)
        dddx = self.calc_dddxt(t)
      else:
        px = last_px

      get_path.append([t, px, dx, ddx, dddx])
    
    return np.array(get_path)