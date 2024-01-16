from enum import Enum
import numpy as np

class AgentType(Enum):
  VEHICLE = 0,
  PEDESTRIAN = 1,

class InterfaceMode(Enum):
  EXTRACT = 0,
  LOAD = 1,

class TrafficLight(Enum):
  UNKOWN = 0,
  RED = 1,
  YELLOW = 2,
  GREEN = 3,
  NONE = 4,

def rgba2rgb(rgba, background=(255,255,255)):
  row, col, ch = rgba.shape
  if ch == 3:
      return rgba
  assert ch == 4, 'RGBA image has 4 channels.'

  rgb = np.zeros( (row, col, 3), dtype='float32' )
  r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

  a = np.asarray( a, dtype='float32' ) / 255.0

  R, G, B = background

  rgb[:,:,0] = r * a + (1.0 - a) * R
  rgb[:,:,1] = g * a + (1.0 - a) * G
  rgb[:,:,2] = b * a + (1.0 - a) * B

  return np.asarray( rgb, dtype='uint8' )

