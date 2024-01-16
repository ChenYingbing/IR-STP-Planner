import os, sys
from tracemalloc import start
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

from utils.kmeans_torch import MyKmeansPytorch

if __name__ == '__main__':
  trajectory_set = np.zeros((10000,2,2), dtype=np.float32)
  initial_k: int = 2
  epsilon_list = [1.0]

  traj_num = trajectory_set.shape[0]
  node_num = trajectory_set.shape[1]
  state_num = trajectory_set.shape[2]

  x_list = [i * 1.0 for i in range(0, node_num)]
  for i in range(traj_num):
    y_list = np.random.randn(node_num, 1) * 1.0
    trajectory_set[i, :, 0] = x_list
    trajectory_set[i, :, 1] = y_list[:, 0]
  
  # points = np.loadtxt('kmeans_demo_data.txt')
  print('Demo trajectory num={}; node_num={}.'.format(traj_num, node_num))
  print('Initial k={}, epsilon_list={}.'.format(initial_k, epsilon_list))

  # data
  start_t = time.time()
  cluster_trajs, center_trajs, cluster_metric = \
    MyKmeansPytorch.run_kmeans(trajectory_set, num_clusters=100, device='cpu')
  dur = time.time() - start_t

  print("Duration={}".format(dur), cluster_metric)

  import matplotlib.pyplot as plt
  fig = plt.figure()

  for o_traj in trajectory_set:
    plt.plot(o_traj[:, 0], o_traj[:, 1], 'b')
  for c_traj in center_trajs:
    plt.plot(c_traj[:, 0], c_traj[:, 1], 'r')

  plt.show()
