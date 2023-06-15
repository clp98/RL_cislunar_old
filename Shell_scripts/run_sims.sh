#!/bin/bash

python main_solve_RL.py --config "config_files/moon_station_keeping_config_COMPUTERONE1.yaml"  #Halo, 4 body, errore, LSTM, max_r=500km, lr costante, espilon=(0.1, 0.001)
python main_solve_RL.py --config "config_files/moon_station_keeping_config_COMPUTERONE2.yaml"  #Halo, 4 body, errore, LSTM, max_r=500km, lr decrescente (-4,-5), epsilon costante
python main_solve_RL.py --config "config_files/moon_station_keeping_config_COMPUTERONE3.yaml"  #Halo, 4 body, errore, LSTM, max_r=500km, lr decrescente(-4,-5), epsilon decrescente
