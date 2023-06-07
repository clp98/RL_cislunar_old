#!/bin/bash

python main_solve_RL.py --config "config_files/moon_station_keeping_config_COMPUTERONE1.yaml"  #NRHO, 4 body, errore, MLP
python main_solve_RL.py --config "config_files/moon_station_keeping_config_COMPUTERONE2.yaml"  #NRHO, 4 body, errore, LSTM
python main_solve_RL.py --config "config_files/moon_station_keeping_config_COMPUTERONE3.yaml"  #NRHO, 4 body, errore, LSTM, max_r=500km
