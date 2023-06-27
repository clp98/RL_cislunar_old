#!/bin/bash

#Halo, 4 body, errore, LSTM, max_r=500km, lr costante, espilon=(0.1, 0.001)
#Halo, 4 body, errore, LSTM, max_r=500km, lr decrescente (-4,-5), epsilon costante

#Halo, 4 body, errore, LSTM, max_r=500km, lr decrescente(-4,-5), epsilon decrescente(0.1,0.001)  [2023-06-16_17-13-17]
#Halo, 4 body, errore, LSTM, max_r=500km, lr decrescente(-4,-5), epsilon decrescente(0.1,0.001), wild_reward [PPO_2023-06-21_15-48-54]

python main_solve_RL_auto.py --config "config_files/moon_station_keeping_config_COMPUTERONE1.yaml"  #4 body, error i.c., max_dist_r, clip=0.05, lr=0.0001 cost, epsilon=(0.1,0.001), MLP
python main_solve_RL_auto.py --config "config_files/moon_station_keeping_config_COMPUTERONE2.yaml"  #4 body, error i.c., max_dist_r, clip=0.05, lr=0.0001 cost, epsilon=(0.1,0.001), LSTM
