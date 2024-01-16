from pyquaternion import Quaternion
import numpy as np
from typing import Dict, Tuple, Union, List
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import math

from preprocessor.preprocess_graph import PreprocessGraphs
from preprocessor.utils import AgentType, TrafficLight
from preprocessor.commonroad.vectorizer import CommonroadVectorizer

import thirdparty.config
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleType
from utils.colored_print import ColorPrinter
from utils.transform import XYYawTransform
from utils.angle_operation import get_normalized_angle

class CommonroadGraphProcessor(PreprocessGraphs, CommonroadVectorizer):
    """
    Class for agent prediction, using the vector representation for maps and agents
    """
    def __init__(self, args: Dict):
      super(PreprocessGraphs, self).__init__(args)
      super(CommonroadVectorizer, self).__init__(args)
      super(CommonroadGraphProcessor, self).__init__(args)

    ### Abstracts
    def get_successor_edges(self, lane_ids: List[Union[str, int]]) -> List[List[int]]:
        """
        Returns successor edge list for each node
        """
        e_succ = []
        giveup_node_num = 0
        for node_id, lane_id in enumerate(lane_ids):
          #@note here node_id means each lane is regarded as a node
          #  for two node in lane_ids may have the same lane id because
          #  one lane from the map may be splited into two lane node.
          e_succ_node = []
          if node_id + 1 < len(lane_ids) and lane_id == lane_ids[node_id + 1]:
            # lane_x = lane_split_seg1 + lane_split_seg2
            e_succ_node.append(node_id + 1)
          else:
            lanelet = self.cache_lanelet_dict[lane_id]
            # print(lanelet.successor)
            # print(lanelet.adj_left, lanelet.adj_left_same_direction)
            # print(lanelet.adj_right, lanelet.adj_right_same_direction)
            for successor_lane_id in lanelet.successor:
              if successor_lane_id in lane_ids:
                e_succ_node.append(lane_ids.index(successor_lane_id))
                # print("edge", lane_id, successor_lane_id)
          
          if len(e_succ) < self.max_nodes:
            e_succ.append(e_succ_node)
          else:
            giveup_node_num += 1
        
        if giveup_node_num > 0:
          ColorPrinter.print('yellow', "number of lanes is larger than {}"
            ", {} nodes are abandoned.".format(self.max_nodes, giveup_node_num))

        # print("get_successor_edges", e_succ)
        return e_succ

