from metrics.metric import Metric
from typing import Dict, Union
import torch
from metrics.utils import miss_rate


class MissRateK(Metric):
    """
    Miss rate for the top K trajectories.
    """

    def __init__(self, args: Dict):
        self.k = args['k']
        self.dist_thresh = args['dist_thresh']
        self.name = 'miss_rate_' + str(self.k)

        self.mask_length: int= 1e+3
        if 'mask_length' in args:
            self.mask_length = int(args['mask_length'])
            self.name = "[{}]".format(self.mask_length) + 'miss_rate_' + str(self.k)

    def compute(self, predictions: Dict, ground_truth: Union[Dict, torch.Tensor]) -> torch.Tensor:
        """
        Compute miss rate
        :param predictions: Dictionary with 'traj': predicted trajectories and 'probs': mode probabilities
        :param ground_truth: Either a tensor with ground truth trajectories or a dictionary
        :return:
        """
        # Unpack arguments
        traj = predictions['traj']
        probs = predictions['probs']
        traj_gt = ground_truth['traj'] if type(ground_truth) == dict else ground_truth

        # Useful params
        batch_size = probs.shape[0]
        num_pred_modes = traj.shape[1]
        sequence_length = traj.shape[2]

        # Masks for variable length ground truth trajectories
        masks = ground_truth['masks'] if type(ground_truth) == dict and 'masks' in ground_truth.keys() \
            else torch.zeros(batch_size, sequence_length).to(traj.device)
        if self.mask_length < sequence_length:
            masks[:, self.mask_length:] = 1.0

        min_k = min(self.k, num_pred_modes)

        _, inds_topk = torch.topk(probs, min_k, dim=1)
        batch_inds = torch.arange(batch_size).unsqueeze(1).repeat(1, min_k)
        traj_topk = traj[batch_inds, inds_topk]

        return miss_rate(traj_topk, traj_gt, masks, dist_thresh=self.dist_thresh)
