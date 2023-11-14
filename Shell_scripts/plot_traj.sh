#!/bin/bash

#indir="results/PPO_2023-05-01_12-25-48/PPO_2023-05-01_13-41-48/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_985ed_00000_0_2023-05-01_13-41-49/checkpoint_000353/"
indir[1]="results/MLP_1halo/PPO_2023-08-04_09-08-29/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_26e19_00000_0_2023-08-04_09-08-29/checkpoint_004886/"
indir[2]="results/LSTM_1halo/PPO_2023-08-07_11-03-36/PPO_2023-08-07_12-02-11/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_e9ef2_00000_0_2023-08-07_12-02-11/checkpoint_004981/"
indir[3]="results/MLP_5halos/PPO_2023-08-03_13-25-53/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_f1a37_00000_0_2023-08-03_13-25-53/checkpoint_004732/"
indir[4]="results/LSTM_5halos/PPO_2023-08-03_14-27-55/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_9c7b0_00000_0_2023-08-03_14-27-56/checkpoint_004718/"
indir[5]="results/MLP_10halos/PPO_2023-08-03_14-38-34/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_190ec_00000_0_2023-08-03_14-38-34/checkpoint_004928/"
indir[6]="results/LSTM_10halos/PPO_2023-08-03_15-14-19/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_17934_00000_0_2023-08-03_15-14-19/checkpoint_004810/"

indir_comp="results/compare/"

for i in {6..6}
do
    # gnuplot -e "indir=\"${indir[$i]}\"" "PlotFiles/plot_traj_3d.plt"
    gnuplot -e "indir=\"${indir[$i]}\"" "PlotFiles/plot_traj_xy.plt"
    gnuplot -e "indir=\"${indir[$i]}\"" "PlotFiles/plot_traj_xz.plt"
    gnuplot -e "indir=\"${indir[$i]}\"" "PlotFiles/plot_traj_yz.plt"
    # gnuplot -e "indir=\"${indir[$i]}\"" "PlotFiles/plot_dist_r.plt"
    # gnuplot -e "indir=\"${indir[$i]}\"" "PlotFiles/plot_dist_v.plt"
    # gnuplot -e "indir=\"${indir[$i]}\"" "PlotFiles/plot_thrust.plt"

    latexmk -pdf
    latexmk -c
    rm *.eps *.tex *-inc-eps-converted-to.pdf
    mv *.pdf ${indir[$i]}
done

gnuplot -e "indir1=\"${indir[1]}\"; indir2=\"${indir[2]}\"; indir3=\"${indir[3]}\"; indir4=\"${indir[4]}\"; indir5=\"${indir[5]}\"; indir6=\"${indir[6]}\";" "PlotFiles/plot_dist_r.plt"
gnuplot -e "indir1=\"${indir[1]}\"; indir2=\"${indir[2]}\"; indir3=\"${indir[3]}\"; indir4=\"${indir[4]}\"; indir5=\"${indir[5]}\"; indir6=\"${indir[6]}\";" "PlotFiles/plot_dist_v.plt"
gnuplot -e "indir1=\"${indir[1]}\"; indir2=\"${indir[2]}\"; indir3=\"${indir[3]}\"; indir4=\"${indir[4]}\"; indir5=\"${indir[5]}\"; indir6=\"${indir[6]}\";" "PlotFiles/plot_thrust.plt"

latexmk -pdf
    latexmk -c
    rm *.eps *.tex *-inc-eps-converted-to.pdf
    mv *.pdf ${indir_comp}
