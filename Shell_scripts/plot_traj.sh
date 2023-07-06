#!/bin/bash

#indir="results/PPO_2023-05-01_12-25-48/PPO_2023-05-01_13-41-48/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_985ed_00000_0_2023-05-01_13-41-49/checkpoint_000353/"
indir="results/PPO_2023-07-03_11-46-17/PPO_2023-07-03_15-26-10/PPO_2023-07-04_20-57-29/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_0fa08_00000_0_2023-07-04_20-57-29/checkpoint_004976/"
gnuplot -e "indir=\"${indir}\"" "PlotFiles/plot_traj.plt"
gnuplot -e "indir=\"${indir}\"" "PlotFiles/plot_traj_2d.plt"

latexmk -pdf
latexmk -c
rm *.eps *.tex *-inc-eps-converted-to.pdf
mv *.pdf ${indir}
