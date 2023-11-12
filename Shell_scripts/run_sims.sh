#!/bin/bash

python main_solve_RL_auto.py --config "config_files/moon_station_keeping_config_polaris1.yaml"  # 1 orbita (51) MLP
python main_solve_RL_auto.py --config "config_files/moon_station_keeping_config_polaris2.yaml"  # 1 orbita (51) LSTM
python main_solve_RL_auto.py --config "config_files/moon_station_keeping_config_polaris3.yaml"  # 101 orbite (1:101) MLP
python main_solve_RL_auto.py --config "config_files/moon_station_keeping_config_polaris4.yaml"  # 101 orbite (1:101) LSTM