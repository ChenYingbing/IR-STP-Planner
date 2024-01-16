#!/bin/bash

export model005_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.05_final'
export model02_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.2_final'
export model04_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.4_final'
export model06_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.6_final'
export model08_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.8_final'
export model10_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp1.0_final'

if [ $# -eq 1 ];
then
  echo "Input with influ_react_min_acc = $1"
  python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 1 --planner_type izone_rela --prediction_mode default --algo_variable3 $1 --acc_mode const --reaction_traj_mode pred --predict_traj_mode_num 1 --solu_tag_str predexp_izone-lrcpred[$1]_1.0_p1;
  wait
  python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 1 --planner_type izone_rela --prediction_mode default --algo_variable3 $1 --acc_mode const --reaction_traj_mode pred --predict_traj_mode_num 2 --solu_tag_str predexp_izone-lrcpred[$1]_1.0_p2;
  wait
  python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 1 --planner_type izone_rela --prediction_mode default --algo_variable3 $1 --acc_mode const --reaction_traj_mode pred --predict_traj_mode_num 3 --solu_tag_str predexp_izone-lrcpred[$1]_1.0_p3;
  wait

  python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 1 --planner_type izone_rela --prediction_mode lon-short --algo_variable3 $1 --acc_mode const --reaction_traj_mode pred --predict_traj_mode_num 2 --solu_tag_str predexp_izone-lrcpredLs[$1]_1.0_p2;
  wait
  python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 1 --planner_type izone_rela --prediction_mode lon-short --algo_variable3 $1 --acc_mode const --reaction_traj_mode pred --predict_traj_mode_num 3 --solu_tag_str predexp_izone-lrcpredLs[$1]_1.0_p3;
  wait

  # python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 1 --planner_type izone_rela --prediction_mode lon-short --algo_variable3 $1 --acc_mode const --reaction_traj_mode pred --predict_traj_mode_num 2 --solu_tag_str exp_izone-lrcpredLs[$1]_1.0_p2;
  # wait
  # python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 1 --planner_type izone_rela --prediction_mode lon-short --algo_variable3 $1 --acc_mode const --reaction_traj_mode pred --predict_traj_mode_num 3 --solu_tag_str exp_izone-lrcpredLs[$1]_1.0_p3;
  # wait

else
  echo "$0: Missing/Too much arguments"
  echo "list of variables= [$@]"
fi
