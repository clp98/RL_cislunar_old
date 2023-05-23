#!/bin/bash

# indir="results/PPO_2023-05-01_12-25-48/PPO_2023-05-01_13-41-48/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_985ed_00000_0_2023-05-01_13-41-49/checkpoint_000353/"
indir="results_done/PPO_2023-05-18_18-50-32/PPO_2023-05-18_22-10-14/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_7032a_00000_0_2023-05-18_22-10-14/checkpoint_001988/"
gnuplot -e "indir=\"${indir}\"" "PlotFiles/plot_traj.plt"
gnuplot -e "indir=\"${indir}\"" "PlotFiles/plot_traj_2d.plt"

latexmk -pdf
latexmk -c
rm *.eps *.tex *-inc-eps-converted-to.pdf
mv *.pdf ${indir}