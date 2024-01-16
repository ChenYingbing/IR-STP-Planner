#!/bin/bash

./sh/run_izone_lrc_irule.sh -0.01 1.0 3.0 -15.0
# var0: -0.01: corresponds to a_i (relation judgement) in paper
# var1: 1.0:   corresponds to c_{f1} (time gap) in paper 
# var2: 3.0:   corresponds to c_{f2} (distance coef) in paper
# var3: -15.0: corresponds to a_i in paper

# in run_izone_lrc_irule.sh or other xxx.sh, further configurations are

# --model_path:
#   the path to the prediction model.

# --is_exp_mode: 
#   0: means all scenarios (232) are tested.
#   1: part scenarios (76) are tested.
#   2: 1 scenarios is tested, for debug only.

# --planner_type:
#   'none_ca':        only forward search without considering collisions with prediction results.
#   'ca':             forward search with navive collision avoidance strategy.
#   'ca+':            same to 'ca', where + means rear prediction results are ignored.
#   'izone_ca':       forward search with overtaking/givingway interaction relations, these 2 are ractive relations.
#   'izone_ca+':      same to 'izone_ca', where + means rear prediction results are ignored.
#   'izone_rela':     forward search with overtaking/givingway/influence interaction relations.
#   'izone_rela+':    same to 'izone_rela', where + means rear prediction results are ignored.

# --acc_mode: useless parameters

# --prediction_mode:
#   default:          all prediction results are considered
#   lon-short:        the most probable one with full time horizon 6.0s / others with 2.0s

# --predict_traj_mode_num: how many modality (int, pred-K) of prediction results all used. Valid values are: 1~10.
