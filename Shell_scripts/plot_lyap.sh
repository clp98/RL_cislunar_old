#!/bin/bash

indir="PO_files/"
gnuplot "PlotFiles/plot_lyap.plt"

latexmk -pdf
latexmk -c
rm *.eps *.tex *-inc-eps-converted-to.pdf
mv *.pdf $indir