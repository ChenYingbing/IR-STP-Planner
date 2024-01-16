import os, sys
import thirdparty.__init__
THIRDPARTY_ROOT = os.path.dirname(thirdparty.__init__.__file__)

sys.path.append(os.path.join(THIRDPARTY_ROOT, "kmeans_pytorch"))

import numpy as np
from kmeans_pytorch import kmeans
import torch
from typing import Dict, List

class MyKmeansPytorch:
  @staticmethod
  def run_kmeans(trajectories: np.ndarray, 
                 num_clusters: int, 
                 device = None):
    '''
    Run kmeans algorithm and return clustered trajectories
    :param trajectories: trajectories in format of (num_traj, num_node, num_state)
    '''
    if device == None:
      # BUG: numpy to torch in 'cuda' seems have some bug, will cause 
      #      to.(device) fails...
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    traj_num = trajectories.shape[0]
    node_num = trajectories.shape[1]
    state_num = trajectories.shape[2]

    cache = trajectories.reshape(traj_num, node_num*state_num)
    process_array = torch.from_numpy(cache)

    # kmeans
    print("If print is not updated, the 'BUG' occurs again. use device = 'cpu'")
    cluster_ids_x, cluster_centers = kmeans(
        X=process_array, num_clusters=num_clusters, 
        distance='euclidean', device=device)

    cluster_ids_x = cluster_ids_x.cpu().numpy() # index: the origin traj is belongs to which cluster
    cluster_centers = cluster_centers.cpu().numpy()

    # get cluster_trajs
    cluster_trajs: Dict[int, List] = {} # map: cluster_idx to list of trajs
    for i, cidx in enumerate(cluster_ids_x):
      if not cidx in cluster_trajs:
        cluster_trajs[cidx] = []
      get_traj = trajectories[i, :, :]
      cluster_trajs[cidx].append(get_traj)

    # get center_trajs
    clust_traj_num = cluster_centers.shape[0]
    clust_node_num = int(cluster_centers.shape[1] / state_num)
    center_trajs = cluster_centers.reshape(
      clust_traj_num, clust_node_num, state_num)

    # get cluster_metric
    cluster_metric: Dict[str, float] = {}
    cluster_metric['max_se'] = 0.0

    for cidx in range(clust_traj_num):
      near_trajs = np.array(cluster_trajs[cidx])
      c_traj = center_trajs[cidx]
      
      # print(near_trajs.shape, c_traj.shape) # (num, 12, 2) (12, 2)
      e_norm = np.linalg.norm(near_trajs-c_traj, 2, 2)
      # print(e_norm.shape) # (18, 12)
      batch_dist_means = np.mean(e_norm, axis=1)
      # print(batch_dist_means.shape) (18, )

      cluster_metric['max_se'] = max(cluster_metric['max_se'], np.max(batch_dist_means))

    return cluster_trajs, center_trajs, cluster_metric

    


