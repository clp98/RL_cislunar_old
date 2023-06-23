#!/bin/bash

#Halo, 4 body, errore, LSTM, max_r=500km, lr costante, espilon=(0.1, 0.001)
#Halo, 4 body, errore, LSTM, max_r=500km, lr decrescente (-4,-5), epsilon costante

#Halo, 4 body, errore, LSTM, max_r=500km, lr decrescente(-4,-5), epsilon decrescente(0.1,0.001)  [2023-06-16_17-13-17]
#Halo, 4 body, errore, LSTM, max_r=500km, lr decrescente(-4,-5), epsilon decrescente(0.1,0.001), wild_reward [PPO_2023-06-21_15-48-54]

python main_solve_RL.py --config "config_files/moon_station_keeping_config.yaml"  #
python main_solve_RL.py --config "config_files/moon_station_keeping_config_2.yaml"  #
