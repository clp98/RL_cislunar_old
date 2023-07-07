#!/bin/bash

#Halo, 4 body, errore, LSTM, max_r=500km, lr costante, espilon=(0.1, 0.001)
#Halo, 4 body, errore, LSTM, max_r=500km, lr decrescente (-4,-5), epsilon costante

#Halo, 4 body, errore, LSTM, max_r=500km, lr decrescente(-4,-5), epsilon decrescente(0.1,0.001)  [2023-06-16_17-13-17]
#Halo, 4 body, errore, LSTM, max_r=500km, lr decrescente(-4,-5), epsilon decrescente(0.1,0.001), wild_reward [PPO_2023-06-21_15-48-54]

#4 body, error i.c., max_dist_r, clip=0.05, lr=0.0001 cost, epsilon=(0.1,0.001), MLP [PPO_2023-06-29_13-48-39]
#4 body, error i.c., max_dist_r, clip=0.05, lr=0.0001 cost, epsilon=(0.1,0.001), LSTM [PPO_2023-06-29_13-48-39]

#max_seq_len: 10 [LSTM_len10_size20]
#lstm_cell_size: 10 [LSTM_len25_size10]

#lmax_seq_len: 35 [LSTM_len35_size20]
#PPO_2023-07-01_21-14-28 con max_seq_len=35, cell_size=10 [LSTM_len35_size10]

python main_solve_RL_auto.py --config "config_files/moon_station_keeping_config_venus1.yaml"  #MLP, 5 halos, fcnet_hiddens: [50,35,20]
python main_solve_RL_auto.py --config "config_files/moon_station_keeping_config_venus2.yaml"  #MLP, 5 halos, fcnet_hiddens: [85,41,20]