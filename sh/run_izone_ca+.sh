#!/bin/bash

export model10_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp1.0_final'

python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 1 --planner_type izone_ca+ --prediction_mode default --predict_traj_mode_num 1 --solu_tag_str exp_izoneca+_1.0_p1;
wait
# python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 1 --planner_type izone_ca+ --prediction_mode default --predict_traj_mode_num 3 --solu_tag_str exp_izoneca+_1.0_p3;
# wait
# python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 1 --planner_type izone_ca+ --prediction_mode default --predict_traj_mode_num 5 --solu_tag_str exp_izoneca+_1.0_p5;
# wait
