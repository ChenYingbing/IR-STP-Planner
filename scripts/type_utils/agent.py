from typing import Dict, List
from utils.transform import XYYawTransform
from type_utils.trajectory_point import TrajectoryPoint
from type_utils.state_trajectory import TrajectoryInfo

AGENT_HUMAN_INDEX: int = 0
AGENT_VEHICLE_INDEX: int = 1

MAP_STR2AGENT_INDEX: Dict[str, Dict[str, int]] = {
  'ca': {
    'vehicle': AGENT_VEHICLE_INDEX,
    'human': AGENT_HUMAN_INDEX,
  },
  'l5kit': {
    'vehicle': AGENT_VEHICLE_INDEX,
    'human': AGENT_HUMAN_INDEX,
  }
}
def string2agent_type(agent_str_type: str, format_str: str='ca') -> int:
  '''
  Transform agent type string to index
  :param agent_str_type: string agent type
  '''
  agent_type_id: int = 1

  if format_str in MAP_STR2AGENT_INDEX.keys():
    agent_type_id = MAP_STR2AGENT_INDEX[format_str][agent_str_type]

  return agent_type_id

def is_vehicle(agent_type_id: int) -> bool:
  return (agent_type_id > 0)

def is_human(agent_type_id: int) -> bool:
  return (agent_type_id == 0)

class EgoAgent:
  def __init__(self, id: int, width: float, length: float, time_step_dt: float) -> None:
    '''
    Class to describe ego agent
    :param time_step_dt: the time invertal (seconds) between two consecutive time steps.
    '''
    self.id = id
    self.info = {
      'id': id,
      'width': width,
      'length': length,
      'time_step_dt': time_step_dt,
    }
    self.states: Dict[int, Dict] = {} # time_step > state records

    self.trajectory_from_time_step: int = 0
    self.trajectory: Dict = {
      'initial_tstep': 0,
      'time_interval': time_step_dt,
      'points': [], # TrajectoryPoint
    }

  def set_state(self, time_step_idx: int, 
                      pos_x: float, pos_y: float, orientation: float,
                      steering_radian: float, 
                      velocity: float, 
                      yaw_rate: float,
                      acceleration: float) -> None:
    '''
    Set state of ego agent, where pos_x, pos_y are position of the vehicle center (rectangle shape)
    :param time_step_idx: time step of the state
    :param velocity: speed of agent m/s
    :param steering_radian: steer of agent (in radian).
    :param yaw_rate: yaw change rate
    '''
    self.states[time_step_idx] = {
      'pos_x': pos_x,
      'pos_y': pos_y,
      'orientation': orientation,
      'steering_radian': steering_radian,
      'velocity': velocity,
      'yaw_rate': yaw_rate,
      'acceleration': acceleration,
    }

  def get_transform(self, time_step_idx: int) -> XYYawTransform:
    '''
    Return xyyaw transform of ego agent at timestep=time_step_idx
    '''
    if not time_step_idx in self.states:
      return None
    
    get_state = self.states[time_step_idx]
    return XYYawTransform(
      x=get_state['pos_x'],
      y=get_state['pos_y'],
      yaw_radian=get_state['orientation']
    )

  # TODO(abing): trajectory operations
