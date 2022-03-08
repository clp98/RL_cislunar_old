#!/bin/bash


gnuplot -e "indir=\"$1\"" "PlotFiles/plot_ctrl.plt"

latexmk -pdf
latexmk -c
rm *.eps *.tex *-inc-eps-converted-to.pdf
mv *.pdf $1