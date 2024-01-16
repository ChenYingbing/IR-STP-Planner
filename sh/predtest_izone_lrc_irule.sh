#!/bin/bash

export model005_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.05_final'
export model02_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.2_final'
export model04_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.4_final'
export model06_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.6_final'
export model08_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp0.8_final'
export model10_path='/home/abing/Workspace/experiment_ws/pyplanning/commonroad/models/net_pgp_gatx2_lvm_pp1.0_final'

if [ $# -eq 4 ];
then
  echo "Input with [constant mode value, ireact_gap_cond_t, ireact_delay_s, influ_react_min_acc] = [$1, $2, $3, $4]"
  python scripts/planning_commonroad.py --model_path ${model005_path} --is_exp_mode 2 --planner_type izone_rela --prediction_mode default --algo_variable1 $2 --algo_variable2 $3 --algo_variable3 $4 --acc_mode const --acc_const_value $1 --reaction_traj_mode irule --predict_traj_mode_num 1 --solu_tag_str predtest_izone-lrcirule$1[$2,$3,$4]_0.05_p1;
  wait
  python scripts/planning_commonroad.py --model_path ${model02_path} --is_exp_mode 2 --planner_type izone_rela --prediction_mode default --algo_variable1 $2 --algo_variable2 $3 --algo_variable3 $4 --acc_mode const --acc_const_value $1 --reaction_traj_mode irule --predict_traj_mode_num 1 --solu_tag_str predtest_izone-lrcirule$1[$2,$3,$4]_0.2_p1;
  wait
  python scripts/planning_commonroad.py --model_path ${model04_path} --is_exp_mode 2 --planner_type izone_rela --prediction_mode default --algo_variable1 $2 --algo_variable2 $3 --algo_variable3 $4 --acc_mode const --acc_const_value $1 --reaction_traj_mode irule --predict_traj_mode_num 1 --solu_tag_str predtest_izone-lrcirule$1[$2,$3,$4]_0.4_p1;
  wait
  python scripts/planning_commonroad.py --model_path ${model06_path} --is_exp_mode 2 --planner_type izone_rela --prediction_mode default --algo_variable1 $2 --algo_variable2 $3 --algo_variable3 $4 --acc_mode const --acc_const_value $1 --reaction_traj_mode irule --predict_traj_mode_num 1 --solu_tag_str predtest_izone-lrcirule$1[$2,$3,$4]_0.6_p1;
  wait
  python scripts/planning_commonroad.py --model_path ${model08_path} --is_exp_mode 2 --planner_type izone_rela --prediction_mode default --algo_variable1 $2 --algo_variable2 $3 --algo_variable3 $4 --acc_mode const --acc_const_value $1 --reaction_traj_mode irule --predict_traj_mode_num 1 --solu_tag_str predtest_izone-lrcirule$1[$2,$3,$4]_0.8_p1;
  wait
  # python scripts/planning_commonroad.py --model_path ${model10_path} --is_exp_mode 2 --planner_type izone_rela --prediction_mode default --algo_variable1 $2 --algo_variable2 $3 --algo_variable3 $4 --acc_mode const --acc_const_value $1 --reaction_traj_mode irule --predict_traj_mode_num 1 --solu_tag_str predtest_izone-lrcirule$1[$2,$3,$4]_1.0_p1;
  # wait

else
  echo "$0: Missing/Too much arguments"
  echo "list of variables= [$@]"
fi
