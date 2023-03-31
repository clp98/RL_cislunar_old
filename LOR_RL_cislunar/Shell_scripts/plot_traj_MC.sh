#!/bin/bash

#indir="sol_saved/PPO_8workers_8envs_100steps_1/"

gnuplot -e "indir=\"$1\"" "PlotFiles/plot_traj_MC.plt"

latexmk -pdf
latexmk -c
rm *.eps *.tex *-inc-eps-converted-to.pdf
mv *.pdf $1