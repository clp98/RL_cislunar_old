#!/bin/bash


i_min=0
i_max=0


dirTR[0]="results/PPO_2022-02-17_18-57-41/"  

dirEXP[0]="results/PPO_2022-02-17_18-57-41/PPO_2022-02-18_11-49-21/PPO_environment.cislunar_env.CislunarEnv_6e249_00000_0_2022-02-18_11-49-21/"

CP[0]=2000

for ((i = i_min ; i <= i_max ; i+=1)); do
    python main_postproc.py --trainer_dir ${dirTR[$i]} --exp_dir ${dirEXP[$i]} --last_cp ${CP[$i]}
done;