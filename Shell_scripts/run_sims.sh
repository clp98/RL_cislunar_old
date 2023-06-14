#!/bin/bash

python main_solve_RL.py --config "config_files/moon_station_keeping_config.yaml"  #Halo, 4 body, errore . LSTM, max_r=500km, espilon=(0.1, 0.001)
python main_solve_RL.py --config "config_files/moon_station_keeping_config_2.yaml"  #NRHO, 4 body, errore, LSTM, max_r=500km, epsilon decrescente(0.1,0.001)
python main_solve_RL.py --config "config_files/moon_station_keeping_config_COMPUTERONE3.yaml"  #NRHO, 4 body, errore, LSTM, max_r=500km, lr decrescente(-4,-5)
