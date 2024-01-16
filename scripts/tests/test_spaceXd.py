from operator import index
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pymath.spaceXd import SpaceXd

space = SpaceXd(
    origins=[0.5, 0.0, 0.0], 
    space_ranges=[[-5.0, 5.0], [-5.0, 5.0], [0.0, 10.0]],
    space_resos=[0.1, 0.1, 0.1],
    allocate_grid_space=True)

space.print_debug_string()

check_inputs = [5.42, 0.0, 0.0]
flag, indexs = space.get_index(check_inputs)

print("check values", check_inputs)
print("indexs", flag, indexs)
print("value", space.get_grid_value(check_inputs))

xyzs = np.array([[0.1, 0.2, 0.3], [0.5, 1.0, 0.5] , [2.0, 3.0, 4.0], [0.0, 4.0, -1.0]])
indexs, valids = space.get_indexs(xyzs)
print("all indexs:", indexs)
print("valids:", valids)
print("valid indexs:", indexs[valids])
