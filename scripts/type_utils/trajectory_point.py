import numpy as np

class TrajectoryPoint:
  def __init__(self, timestamp: float=0.0, 
                     pos_x: float=0.0, pos_y: float=0.0, pos_yaw: float=0.0,
                     steer: float=0.0, velocity: float=0.0, 
                     acceleration: float = 0.0):
    '''
    :param pos_x/pos_y: in m
    :param pos_yaw: in radian
    :param velocity: in m/s
    :param timestamp: in seconds
    '''
    self._state = None
    self.set_values(
      timestamp, pos_x, pos_y, pos_yaw, steer, velocity, acceleration)

  def set_values(self, timestamp: float=0.0,
                       pos_x: float=0.0, pos_y: float=0.0, pos_yaw: float=0.0,
                       steer: float=0.0, velocity: float=0.0, 
                       acceleration: float = 0.0):
    self._state = [
      timestamp, 
      pos_x, pos_y, pos_yaw, 
      steer, velocity, 
      acceleration]

  def get_xy_array(self):
    #@note dtype=np.float64 is to support np.sqrt calculation
    return np.array(self._state, dtype=np.float64)[1:3]

  def state(self):
    return {
      'timestamp': self._state[0],
      'pos_x': self._state[1],
      'pos_y': self._state[2],
      'pos_yaw': self._state[3],
      'steer': self._state[4],
      'velocity': self._state[5],
      'acceleration': self._state[6],
    }

  @staticmethod
  def create_point(pos_x: float=0.0, pos_y: float=0.0, pos_yaw: float=0.0):
    '''
    Return a TrajectoryPoint given x, y, yaw
    '''
    return TrajectoryPoint(pos_x=pos_x, pos_y=pos_y, pos_yaw=pos_yaw)
