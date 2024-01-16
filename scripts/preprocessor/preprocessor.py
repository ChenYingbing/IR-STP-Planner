import abc
import torch.utils.data as torch_data
import numpy as np
from typing import Union, Dict
import os

from preprocessor.utils import TrafficLight

class PreprocessInterface:
    """
    Base class for trajectory datasets.
    """
    def __init__(self):
        """
        Initialize trajectory dataset.
        """
        super(PreprocessInterface, self).__init__()
        self.poly_priority_idx: Dict[str, float] = {
         'ped_crossing': 1.0,
         'stop_line': 3.0,
        }

        self.tl_color_to_priority_idx: Dict[TrafficLight, float] = {
            TrafficLight.UNKOWN: 0.0, 
            TrafficLight.GREEN: 1.0, 
            TrafficLight.YELLOW: 2.0, 
            TrafficLight.RED: 3.0, 
            TrafficLight.NONE: 4.0
        }

    ### Port functions
    def process(self, idx: int, agent_idx: int):
        """
        Function to process data.
        :param idx: data index
        :param agent_idx: agent index
        """
        inputs = self.get_inputs(idx, agent_idx)
        ground_truth = self.get_ground_truth(idx, agent_idx)
        data = {'inputs': inputs, 'ground_truth': ground_truth}

        return data

    ### Functions
    def get_inputs(self, idx: int, agent_idx: int) -> Dict:
        """
        Gets model inputs for agent prediction
        :param idx: data index
        :param agent_idx: agent index
        :return inputs: Dictionary with input representations
        """
        map_representation =\
            self.extract_map_representation(idx, agent_idx)
        surrounding_agent_representation =\
            self.extract_surrounding_agent_representation(idx, agent_idx)
        target_agent_representation =\
            self.extract_target_agent_representation(idx, agent_idx)
        inputs = {'data_index': idx,
                  'agent_index': agent_idx,
                  'map_representation': map_representation,
                  'surrounding_agent_representation': surrounding_agent_representation,
                  'target_agent_representation': target_agent_representation}

        return inputs

    def get_ground_truth(self, idx: int, agent_idx: int) -> Dict:
        """
        Extracts ground truth 'labels' for training.
        :param idx: data index
        :param agent_idx: agent index
        """
        target_future = self.extract_target_representation(idx, agent_idx)
        ground_truth = {'traj': target_future}

        return ground_truth

    ### Extractions
    @abc.abstractmethod
    def extract_target_agent_representation(self, idx: int, agent_idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts target agent representation
        :param idx: data index
        :param agent_idx: agent index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_map_representation(self, idx: int, agent_idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts map representation
        :param idx: data 
        :param agent_idx: agent index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_surrounding_agent_representation(self, idx: int, agent_idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts surrounding agent representation
        :param idx: data index
        :param agent_idx: agent index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_target_representation(self, idx: int, agent_idx: int) -> Union[np.ndarray, Dict]:
        """
        Extracts target representation for target agent
        :param idx: data index
        :param agent_idx: agent index
        """
        raise NotImplementedError()    

