import numpy as np
import math
from typing import Tuple

__grid_t_reso = 0.5
__grid_t_reso_2 = __grid_t_reso * 0.5

def assert_dict_name_is_legal(dict_name: str):
  '''
  assert that the dict name is legal
  '''
  assert dict_name in __grid_dicts.keys(), "Error, {} not in {}.".format(dict_name, __grid_dicts.keys())

def get_reaction_acc_value(dict_name: str,
                           agent_i_v0: float, agent_i_move_s: float,
                           agent_j_v0: float, agent_j_move_s: float,
                           default_acc: float = 0.0) -> float:
  '''
  return expected reaction acc value of agent i
  :param dict_name: name of the grid dict, which format follows the paramters in data collection
  :param agent_i_v0: initial speed of agent i when facing the potential interaction (point)
  :param agent_i_move_s: agent i's travelling distance to the interaction point
  :param agent_j_v0: initial speed of agent j when facing the potential interaction (point)
  :param agent_j_move_s: agent j's travelling distance to the interaction point
  :param default_acc: default acc value when math.math.math.nan value is recorded in the __grid_dict
  '''
  min_cal_v0 = 0.25
  max_reach_t = 10.0

  i_reach_t = min(max_reach_t, agent_i_move_s / max(agent_i_v0, min_cal_v0))
  j_reach_t = min(max_reach_t, agent_j_move_s / max(agent_j_v0, min_cal_v0))

  xid :int = math.floor((i_reach_t - __x_axis_info['vmin'] + __grid_t_reso_2) / __grid_t_reso)
  yid :int = math.floor((j_reach_t - __y_axis_info['vmin'] + __grid_t_reso_2) / __grid_t_reso)

  if not (xid, yid) in __grid_dicts[dict_name].keys():
    return default_acc
  return __grid_dicts[dict_name][xid, yid]

def get_reaction_acc_values(dict_name: str,
                            agent_i_v0: np.ndarray, agent_i_move_s: np.ndarray,
                            agent_j_v0: float, agent_j_move_s: np.ndarray,
                            default_acc: float = 0.0) -> float:
  '''
  return expected reaction acc values of agent i
  :param dict_name: name of the grid dict, which format follows the paramters in data collection
  :param agent_i_v0: initial speed of agent i when facing the potential interaction (point)
  :param agent_i_move_s: agent i's travelling distance to the interaction point
  :param agent_j_v0: initial speed of agent j when facing the potential interaction (point)
  :param agent_j_move_s: agent j's travelling distance to the interaction point
  :param default_acc: default acc value when math.math.math.nan value is recorded in the __grid_dict
  '''
  min_cal_v0 = 0.25
  max_reach_t = 10.0

  # protect v0's values
  _agent_i_v0 = agent_i_v0.copy()
  _agent_i_v0[_agent_i_v0 < min_cal_v0] = min_cal_v0
  _agent_j_v0 = max(agent_j_v0, min_cal_v0)

  i_reach_t = agent_i_move_s / _agent_i_v0
  j_reach_t = agent_j_move_s / _agent_j_v0
  i_reach_t[i_reach_t > max_reach_t] = max_reach_t
  j_reach_t[j_reach_t > max_reach_t] = max_reach_t

  xids = np.floor((i_reach_t - __x_axis_info['vmin'] + __grid_t_reso_2) / __grid_t_reso)
  yids = np.floor((j_reach_t - __y_axis_info['vmin'] + __grid_t_reso_2) / __grid_t_reso)

  get_accs = []
  for xid, yid in zip(xids, yids):
    if not (xid, yid) in __grid_dicts[dict_name]:
      get_accs.append(default_acc)
    elif math.isnan(__grid_dicts[dict_name][xid, yid]):
      get_accs.append(default_acc)
    else:
      get_accs.append(__grid_dicts[dict_name][xid, yid])
  return np.array(get_accs)

__x_axis_info = {'vmin': 0.005480663218388766, 'vmax': 10.0, 'idmin': 0, 'idmax': 20}
__y_axis_info = {'vmin': 0.005999085161820012, 'vmax': 10.0, 'idmin': 0, 'idmax': 20}

__x_axis_siz = int(__x_axis_info['idmax']-__x_axis_info['idmin'] + 1)
__y_axis_siz = int(__y_axis_info['idmax']-__y_axis_info['idmin'] + 1)

# scene_min_num: 10
# boostrap sampling, confidence: 0.95
# prob_condition: 0.8
#
# Inputs:   grid x, y are t values of expected arrival time of agents
# Output:   acc value to overtake (> 0) / giveway (< 0) / or unkown (== math.math.math.nan)
__grid_dicts = {
  '[s10p0.8]': {(0, 0): math.nan, (1, 0): math.nan, (2, 0): math.nan, (3, 0): -1.2311058285296577, (4, 0): -1.5022247982237844, (5, 0): -1.1504676595393348, (6, 0): -0.4191772632418377, (7, 0): -0.8402316675663519, (8, 0): math.nan, (9, 0): math.nan, (11, 0): math.nan, (12, 0): math.nan, (13, 0): math.nan, (14, 0): math.nan, (15, 0): math.nan, (16, 0): math.nan, (17, 0): math.nan, (18, 0): math.nan, (19, 0): math.nan, (20, 0): math.nan, (0, 1): math.nan, (1, 1): math.nan, (2, 1): math.nan, (3, 1): -1.5520574822319277, (4, 1): -1.2768637992151033, (5, 1): -1.0612774731684473, (6, 1): -1.2473084307822955, (7, 1): -0.8206113818481011, (8, 1): -0.41657592423135675, (9, 1): math.nan, (11, 1): math.nan, (12, 1): math.nan, (13, 1): math.nan, (14, 1): math.nan, (15, 1): math.nan, (16, 1): math.nan, (17, 1): math.nan, (18, 1): math.nan, (19, 1): math.nan, (20, 1): math.nan, (0, 2): math.nan, (1, 2): math.nan, (2, 2): math.nan, (3, 2): math.nan, (4, 2): -1.9749738364138294, (5, 2): -1.0934052809666754, (6, 2): -0.970107348431099, (7, 2): -1.043694324842716, (8, 2): -0.5419681904496407, (9, 2): math.nan, (11, 2): math.nan, (12, 2): math.nan, (13, 2): math.nan, (14, 2): math.nan, (15, 2): math.nan, (16, 2): math.nan, (17, 2): math.nan, (18, 2): math.nan, (19, 2): math.nan, (20, 2): math.nan, (0, 3): 1.4513583333333333, (1, 3): 1.1650390277777762, (2, 3): 1.7543716666666669, (3, 3): math.nan, (4, 3): math.nan, (5, 3): -1.3866600664340456, (6, 3): -0.950868470474374, (7, 3): -0.8884720913091276, (8, 3): math.nan, (9, 3): math.nan, (11, 3): math.nan, (12, 3): math.nan, (13, 3): math.nan, (14, 3): math.nan, (15, 3): math.nan, (16, 3): math.nan, (17, 3): math.nan, (18, 3): math.nan, (19, 3): math.nan, (20, 3): math.nan, (0, 4): 1.4458573611111103, (1, 4): 1.3961700992063493, (2, 4): 1.533388703703704, (3, 4): -1.3021977742705824, (4, 4): math.nan, (5, 4): math.nan, (6, 4): math.nan, (7, 4): -0.8312216110555408, (8, 4): -0.7429907457312138, (9, 4): math.nan, (11, 4): math.nan, (12, 4): math.nan, (13, 4): math.nan, (14, 4): math.nan, (15, 4): math.nan, (16, 4): math.nan, (17, 4): math.nan, (18, 4): math.nan, (19, 4): math.nan, (20, 4): math.nan, (0, 5): 1.473324781746033, (1, 5): 1.1939955720899476, (2, 5): 1.0251980660080655, (3, 5): 1.663531558512294, (4, 5): math.nan, (5, 5): math.nan, (6, 5): math.nan, (7, 5): math.nan, (8, 5): math.nan, (9, 5): math.nan, (11, 5): math.nan, (12, 5): math.nan, (13, 5): math.nan, (14, 5): math.nan, (15, 5): math.nan, (16, 5): math.nan, (17, 5): math.nan, (18, 5): math.nan, (19, 5): math.nan, (20, 5): math.nan, (0, 6): 1.5885089682539673, (1, 6): 1.3408117679773932, (2, 6): 1.154846143023643, (3, 6): 1.098286037940817, (4, 6): 1.247146954191033, (5, 6): math.nan, (6, 6): math.nan, (7, 6): math.nan, (8, 6): math.nan, (9, 6): math.nan, (11, 6): math.nan, (12, 6): math.nan, (13, 6): math.nan, (14, 6): math.nan, (15, 6): math.nan, (16, 6): math.nan, (17, 6): math.nan, (18, 6): math.nan, (19, 6): math.nan, (20, 6): math.nan, (0, 7): 1.1422603174603172, (1, 7): 1.2841628333934585, (2, 7): 1.317185284021534, (3, 7): 1.123703905322869, (4, 7): 0.8768785174811204, (5, 7): 0.57828874546599, (6, 7): math.nan, (7, 7): math.nan, (8, 7): math.nan, (9, 7): math.nan, (11, 7): math.nan, (12, 7): math.nan, (13, 7): math.nan, (14, 7): math.nan, (15, 7): math.nan, (16, 7): math.nan, (17, 7): math.nan, (18, 7): math.nan, (19, 7): math.nan, (20, 7): math.nan, (0, 8): 1.4905262500000014, (1, 8): 1.384808759920635, (2, 8): 1.2248469594512241, (3, 8): 1.1565676941968113, (4, 8): 1.087689518959182, (5, 8): 1.2089605823471348, (6, 8): math.nan, (7, 8): math.nan, (8, 8): math.nan, (9, 8): math.nan, (11, 8): math.nan, (12, 8): math.nan, (13, 8): math.nan, (14, 8): math.nan, (15, 8): math.nan, (16, 8): math.nan, (17, 8): math.nan, (18, 8): math.nan, (19, 8): math.nan, (20, 8): math.nan, (0, 9): 1.7368684722222214, (1, 9): 1.4677037957875454, (2, 9): 1.2688380370827437, (3, 9): 1.0980340077793938, (4, 9): 1.2322847405819408, (5, 9): 1.0402565900565486, (6, 9): 1.0420033100497157, (7, 9): math.nan, (8, 9): math.nan, (9, 9): math.nan, (11, 9): math.nan, (12, 9): math.nan, (13, 9): math.nan, (14, 9): math.nan, (15, 9): math.nan, (16, 9): math.nan, (17, 9): math.nan, (18, 9): math.nan, (19, 9): math.nan, (20, 9): math.nan, (0, 11): 1.847134117063492, (1, 11): 1.7581753090659331, (2, 11): math.nan, (3, 11): math.nan, (4, 11): math.nan, (5, 11): 0.6205118079284508, (6, 11): math.nan, (7, 11): math.nan, (8, 11): 1.1063517813485966, (9, 11): math.nan, (11, 11): math.nan, (12, 11): math.nan, (13, 11): math.nan, (14, 11): math.nan, (15, 11): math.nan, (16, 11): math.nan, (17, 11): math.nan, (18, 11): math.nan, (19, 11): math.nan, (20, 11): math.nan, (0, 12): 1.9390727777777785, (1, 12): 1.8120761011904767, (2, 12): 1.4451700777925773, (3, 12): 1.5916833251035456, (4, 12): 1.284990861579368, (5, 12): 1.3508742920783061, (6, 12): 1.4842087243328579, (7, 12): 1.257437290760978, (8, 12): 1.7141221037348304, (9, 12): 1.2610293868740419, (11, 12): math.nan, (12, 12): math.nan, (13, 12): math.nan, (14, 12): math.nan, (15, 12): math.nan, (16, 12): math.nan, (17, 12): math.nan, (18, 12): math.nan, (19, 12): math.nan, (20, 12): math.nan, (0, 13): math.nan, (1, 13): 1.8959972481684981, (2, 13): 1.591135886058386, (3, 13): 1.667306876381824, (4, 13): 1.5927435311373352, (5, 13): 1.6086762730635424, (6, 13): 1.487108297250211, (7, 13): 1.1608884273619775, (8, 13): 1.382227412870003, (9, 13): math.nan, (11, 13): math.nan, (12, 13): math.nan, (13, 13): math.nan, (14, 13): math.nan, (15, 13): math.nan, (16, 13): math.nan, (17, 13): math.nan, (18, 13): math.nan, (19, 13): math.nan, (20, 13): math.nan, (0, 14): math.nan, (1, 14): 1.9244554969336218, (2, 14): math.nan, (3, 14): 1.8316555503034544, (4, 14): 1.6411245770983838, (5, 14): 1.69303809811887, (6, 14): math.nan, (7, 14): math.nan, (8, 14): 1.5088096783852698, (9, 14): math.nan, (11, 14): math.nan, (12, 14): math.nan, (13, 14): math.nan, (14, 14): math.nan, (15, 14): math.nan, (16, 14): math.nan, (17, 14): math.nan, (18, 14): math.nan, (19, 14): math.nan, (20, 14): math.nan, (0, 15): math.nan, (1, 15): 1.9199474816849813, (2, 15): 1.3898992124542127, (3, 15): math.nan, (4, 15): math.nan, (5, 15): 1.862205477327624, (6, 15): math.nan, (7, 15): math.nan, (8, 15): math.nan, (9, 15): math.nan, (11, 15): math.nan, (12, 15): math.nan, (13, 15): math.nan, (14, 15): math.nan, (15, 15): math.nan, (16, 15): math.nan, (17, 15): math.nan, (18, 15): math.nan, (19, 15): math.nan, (20, 15): math.nan, (0, 16): math.nan, (1, 16): 1.7009786696174196, (2, 16): math.nan, (3, 16): math.nan, (4, 16): 1.6962116737550947, (5, 16): 1.6185295665264687, (6, 16): math.nan, (7, 16): math.nan, (8, 16): math.nan, (9, 16): math.nan, (11, 16): math.nan, (12, 16): math.nan, (13, 16): math.nan, (14, 16): math.nan, (15, 16): math.nan, (16, 16): math.nan, (17, 16): math.nan, (18, 16): math.nan, (19, 16): math.nan, (20, 16): math.nan, (0, 17): math.nan, (1, 17): math.nan, (2, 17): math.nan, (3, 17): math.nan, (4, 17): math.nan, (5, 17): 1.667723143096034, (6, 17): math.nan, (7, 17): math.nan, (8, 17): math.nan, (9, 17): math.nan, (11, 17): math.nan, (12, 17): math.nan, (13, 17): math.nan, (14, 17): math.nan, (15, 17): math.nan, (16, 17): math.nan, (17, 17): math.nan, (18, 17): math.nan, (19, 17): math.nan, (20, 17): math.nan, (0, 18): math.nan, (1, 18): math.nan, (2, 18): math.nan, (3, 18): math.nan, (4, 18): math.nan, (5, 18): math.nan, (6, 18): math.nan, (7, 18): math.nan, (8, 18): math.nan, (9, 18): math.nan, (11, 18): math.nan, (12, 18): math.nan, (13, 18): math.nan, (14, 18): math.nan, (15, 18): math.nan, (16, 18): math.nan, (17, 18): math.nan, (18, 18): math.nan, (19, 18): math.nan, (20, 18): math.nan, (0, 19): math.nan, (1, 19): math.nan, (2, 19): math.nan, (3, 19): math.nan, (4, 19): 1.6661195520254888, (5, 19): math.nan, (6, 19): math.nan, (7, 19): math.nan, (8, 19): math.nan, (9, 19): math.nan, (11, 19): math.nan, (12, 19): math.nan, (13, 19): math.nan, (14, 19): math.nan, (15, 19): math.nan, (16, 19): math.nan, (17, 19): math.nan, (18, 19): math.nan, (19, 19): math.nan, (20, 19): math.nan, (0, 20): 1.7968591269841272, (1, 20): 1.636162471493648, (2, 20): 1.647272850425718, (3, 20): 1.6214154193160986, (4, 20): 1.6610801392863601, (5, 20): math.nan, (6, 20): 1.6422551555774103, (7, 20): math.nan, (8, 20): 1.5116136679325964, (9, 20): math.nan, (11, 20): 1.6764485888497456, (12, 20): math.nan, (13, 20): math.nan, (14, 20): math.nan, (15, 20): math.nan, (16, 20): math.nan, (17, 20): math.nan, (18, 20): math.nan, (19, 20): math.nan, (20, 20): math.nan},
}
