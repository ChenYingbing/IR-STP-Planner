from metrics.metric import Metric
from typing import Dict, Union
import torch


class BinaryCrossEntropyLoss(Metric):
    """
    Metric class which packs the torch.nn.BCELoss()
    """
    def __init__(self, args: Dict):
        self.name = 'bce_loss'

        self.bce_loss = torch.nn.BCELoss()

    def compute(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Compute nn.bce loss

        :param predictions: prediction probabilities of a classification task
        :param ground_truth: ground truth classificiation results
        """        
        # print("ground_truth", predictions.shape, ground_truth.shape)
        # print(ground_truth[:, 0])
        return self.bce_loss(predictions, ground_truth)
