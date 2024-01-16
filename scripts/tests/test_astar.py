import os, sys
from sklearn import neighbors
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pymath.astar import AStarSearch

astar_utils = AStarSearch()

start = (0, 0)
goal = (5, 5)
neighbors_list = {
  (0, 0): [(1, 0), (2, 0), (3, 0), (10, 0)],
  (1, 0): [(0, 1), (2, 1)],
  (2, 0): [(4, 1), (3, 1)],
  (3, 0): [],
  (10, 0): [],
  (0, 1): [(1, 5), (2, 5)],
  (2, 1): [(3, 5)],
  (4, 1): [(4, 5)],
  (3, 1): [(5, 5)],
  (1, 5): [],
  (2, 5): [],
  (3, 5): [],
  (4, 5): [],
  (5, 5): [],
}

astar_utils.ReInit()

astar_utils.Add2Openlist(None, start, 0.0, 0.0) # add start node

for i in range(0, 1000):
  is_empty, parent_index = astar_utils.PopOpenList()
  print(i, "is_empty={}".format(is_empty))
  if is_empty:
    break
  
  parent_node = astar_utils.get_node(parent_index)

  print("current_node_index=", parent_index)
  # add neighbour nodes
  for child_index in neighbors_list[parent_index]:
    g = parent_node.g + 1.0
    astar_utils.Add2Openlist(parent_index, child_index, g=g, h=0.0)

get_path = astar_utils.ExtractPath(maximum_iterations=1000, start_index=start, goal_index=goal)
print("get_path", get_path)
