import math
from typing import List, Dict, Any

class AABBBox:
  def __init__(self, x_bounds: List, y_bounds: List) -> None:
    self.set_values(x_bounds, y_bounds)

  def set_values(self, x_bounds: List, y_bounds: List) -> None:
    self.x_min = x_bounds[0]
    self.x_max = x_bounds[1]
    self.y_min = y_bounds[0]
    self.y_max = y_bounds[1]

  def bounds(self) -> List:
    return [self.x_min, self.x_max, self.y_min, self.y_max]

  @staticmethod
  def overlapped(box1, box2) -> bool:
    not_overlap = ((box1.x_max < box2.x_min) or (box1.x_min > box2.x_max)) and\
                  ((box1.y_max < box2.y_min) or (box1.y_min > box2.y_max))
    return not not_overlap  

class PolygonList:
  aabb: AABBBox = None
  poly_list: List[Dict[str, Any]] = []
