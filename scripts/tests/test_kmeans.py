import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List
from utils.kmeans import TrajectoryKmeans

if __name__ == '__main__':
    trajectory_set = np.zeros((1000,12,2), dtype=np.float32)
    initial_k: int = 2
    epsilon_list = [1.0]

    traj_num = trajectory_set.shape[0]
    node_num = trajectory_set.shape[1]
    
    x_list = [i * 1.0 for i in range(0, node_num)]
    for i in range(traj_num):
      y_list = np.random.randn(node_num, 1) * 1.0
      trajectory_set[i, :, 0] = x_list
      trajectory_set[i, :, 1] = y_list[:, 0]
    
    # points = np.loadtxt('kmeans_demo_data.txt')
    print('Demo trajectory num={}; node_num={}.'.format(traj_num, node_num))
    print('Initial k={}, epsilon_list={}.'.format(initial_k, epsilon_list))
    tkmeans = TrajectoryKmeans()

    last_k: int = 0
    traj_clustered: List = []
    last_centroids: List = []
    for epsilon in epsilon_list:
      max_se:float = 1e+4
      k: int = max(initial_k, last_k)
      while max_se > epsilon:
        # Starting to find clustered traj with respect to epsilon and k.
        if k > traj_num:
          print("\nQuit to find k trajectories since the k={} > traj_num={}.".format(k, traj_num))
          break
        traj_clustered, last_centroids, max_se = \
          tkmeans.kmeans_max_se(trajectory_set, k=k, max_iter=100, epochs=3, epsilon=epsilon, verbose=True)      
        k = len(last_centroids) * 2
 
      last_k = k

    # clusters, centroids, max_ses = tkmeans.kmeans_max_se(
    #     points=trajectory_set, k=initial_k, epochs=10, 
    #     max_iter=1000, epsilon=epsilon, verbose=True)
    print('\nResult clusters num={}, centroids num={}.'.format(
      len(traj_clustered), len(last_centroids)))
    # for cluster, centroid in zip(clusters, centroids):
    #   print(cluster.shape)  # (6, 12, 2), (4, 12, 2): trajectories for a centroid
    #   print(centroid.shape) # (12, 2)

    tkmeans.visualize_traj_clusters(traj_clustered, last_centroids)
