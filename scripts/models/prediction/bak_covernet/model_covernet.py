import os
import torch
import numpy as np
from typing import List, Dict

from utils.file_io import extract_folder_file_list
from models.prediction.model import BasicPredictionModel
from utils.transform import XYYawTransform

class PredictionModelCovernet(BasicPredictionModel):
  def __init__(self, device:str, model_path: str, 
                     predict_traj_num: int = 10,
                     thres_traj_sum_prob: float = 0.95,
                     thres_traj_min_prob: float = 0.03):
    '''
    :param predict_traj_num: the least amount the trajectories being picked up
    :param thres_traj_sum_prob: remove trajs if sum of trajectory probabilities is >= thres_traj_sum_prob
    :param thres_traj_min_prob: remove trajs if its probabilities < thres_traj_min_prob
    '''
    BasicPredictionModel.__init__(self, device, model_path)

    self.netmodel = torch.load(model_path)
    self.netmodel = self.netmodel.to(device)

    self.predict_traj_num = predict_traj_num
    self.thres_traj_sum_prob = thres_traj_sum_prob
    self.thres_traj_min_prob = thres_traj_min_prob

    # extract epsilon_file from log_dir
    log_dir = os.path.dirname(model_path)
    file_list = extract_folder_file_list(log_dir)
    epsilon_file_list = []
    for f in file_list:
      if 'epsilon' in f:
        epsilon_file_list.append(f)
    assert len(epsilon_file_list) == 1, 'Error, fails to find epsilon file from {}'.format(log_dir)

    self.lattice_trajs = np.load(os.path.join(log_dir, epsilon_file_list[0]))
    assert (self.lattice_trajs.ndim == 3), 'Error, unkown lattice traj shape={}.'.format(self.lattice_trajs.shape)
    assert (self.lattice_trajs.shape[2] == 2), 'Error, unkown lattice traj shape={}.'.format(self.lattice_trajs.shape)

    self.lattice_trajs_num = self.lattice_trajs.shape[0]
    assert predict_traj_num < self.lattice_trajs_num,\
      'Error, value of predict_traj_num={}/{} is too large.'.format(predict_traj_num, self.lattice_trajs_num)

  def predict(self, ego_xyyaw: XYYawTransform,
                    agent_list: List[Dict],
                    agent_states: List[XYYawTransform],
                    inputs: List) -> List:
    '''
    Return list of agent predicted trajectories.
    '''
    agent_tensor = inputs['state_tensor']
    image_tensor = inputs['image_tensor']

    # print("predict", agent_tensor.shape, image_tensor.shape)
    agent_tensor = agent_tensor.float().to(self.device)
    image_tensor = image_tensor.float().to(self.device)
    image_tensor /= 255.0 # normalization.
    input_batch_size = agent_tensor.shape[0]
    assert len(agent_states) == input_batch_size,\
      'Error, the size is unmatched {}/{}.'.format(len(agent_states), input_batch_size)

    batch_size = min(input_batch_size, self.predict_batch_siz)
    batchs = list(range(0, input_batch_size, batch_size))
    agents_trajs = []
    agent_id = 0
    for bid in batchs:
      input1 = image_tensor[bid:(bid+batch_size), :]
      input2 = agent_tensor[bid:(bid+batch_size), :]
      output = self.netmodel(input1, input2)
      # print(bid, output.shape) # [batch_size, lattice_traj_num]
      assert (output.shape[1] == self.lattice_trajs_num), \
        'Error, traj size mismatch {}/{}.'.format(output.shape[1], self.lattice_trajs_num)

      probs = torch.softmax(output, dim=1)
      batch_indexs =\
        probs.argsort(descending=True)[:, 0:self.predict_traj_num].cpu().detach().numpy() # [5, 10]

      for bbid in range(batch_indexs.shape[0]):
        # each batch is prediction of a target agent
        agent_state = agent_states[agent_id]
        # print("agent[{}] = {}".format(agent_id, agent_state._x, agent_state._y))

        trajs_list = []
        sum_prob = 0.0
        for traj_index in batch_indexs[bbid, :]:
          prob = probs[bbid, traj_index].cpu().detach().item()
          if prob < self.thres_traj_min_prob:
            continue
          if sum_prob >= self.thres_traj_sum_prob:
            break

          rela_pred_traj = self.lattice_trajs[int(traj_index), :, :]
          pred_traj = self.transform2global_frame(agent_state, rela_pred_traj, True)
          pred_traj = self._estimate_trajectory_yaws(agent_state, pred_traj)

          # print("pred_traj", pred_traj.shape) # [12, 3]
          trajs_list.append({'prob': prob, 'trajectory': pred_traj})

          sum_prob += prob

        # reweighted
        for trajinfo in trajs_list:
          trajinfo['prob'] /= sum_prob

        agents_trajs.append(trajs_list) # one agent contains many trajectories
        agent_id += 1

    assert (len(agents_trajs) == input_batch_size),\
      "Error, BUG happens {}/{}".format(len(agents_trajs), input_batch_size)

    return agents_trajs
