#!/bin/bash


i_min=0
i_max=0


dirTR[0]="A1/MLP/PPO_20workers_3envs_1000steps_1/"  

dirEXP[0]="PPO_2021-10-02_12-17-26/PPO_NeosEnv_f1596_00000_0_2021-10-02_12-17-27/"

CP[0]="1000"

dirCP[0]="checkpoint_0001000/checkpoint-1000"

for ((i = i_min ; i <= i_max ; i+=1)); do
    python main_graphs.py --trainer_dir "sol_completed/"${dirTR[$i]} --exp_dirs "sol_completed/"${dirTR[$i]}${dirEXP[$i]} --last_cps ${CP[$i]} --best_cp_dir "sol_completed/"${dirTR[$i]}${dirEXP[$i]}${dirCP[$i]}
done;