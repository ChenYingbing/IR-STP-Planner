from planners.planner import BasicPlanner
from typing import List, Dict
import numpy as np
import math

class SpeedProfileGenerator(BasicPlanner):
  '''
  Motion planning algorithm based on interaction point model.
  '''
  def __init__(self, plan_horizon_T: float, 
               predict_traj_mode_num: int,
               planner_type:str='ca',
               reaction_config:Dict=None):
    BasicPlanner.__init__(self, 
      planner_type=planner_type,
      plan_horizon_T=plan_horizon_T,
      predict_traj_mode_num=predict_traj_mode_num,
      avoid_d_range=[0., 0.],
      reaction_config=reaction_config)
