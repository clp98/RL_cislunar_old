#!/bin/bash


i_min=0
i_max=0


dirTR[0]="results/PPOTrainer_2022-03-08_16-41-43/"  

dirEXP[0]="results/PPOTrainer_2022-03-08_16-41-43/PPOTrainer_environment.cislunar_env.CislunarEnv_41033_00000_0_2022-03-08_16-41-43/"

CP[0]=497

for ((i = i_min ; i <= i_max ; i+=1)); do
    python main_postproc.py --trainer_dir ${dirTR[$i]} --exp_dir ${dirEXP[$i]} --last_cp ${CP[$i]}
done;