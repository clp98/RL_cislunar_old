
filename1 = indir1.'theta_dist.txt'
filename2 = indir2.'theta_dist.txt'
filename3 = indir3.'theta_dist.txt'
filename4 = indir4.'theta_dist.txt'
filename5 = indir5.'theta_dist.txt'
filename6 = indir6.'theta_dist.txt'


set term unknown

set key top left nobox opaque #height 0.5 width 1.3
set grid 
# unset key

# set xlabel "$t/t_{\\text{max}}$"
set xlabel "$\\beta,\\, \\si{deg}$"
set ylabel "$d_r/\\|\\bm{r}_{\\mathcal{H}}\\|,\\, \\%$"
set xrange [-180:180]
set xtics 45
set yrange [0:0.04]
# set ytics 0, 25, 150

set format x "%.0f"
set format y "%.3f"

set style line 1 lt 6 lw 2 pt 7 pi -1 ps 1.5
set style line 2 lt 6 lw 2 pt 9 pi -1 ps 1.5
set style line 3 lt 7 lw 2 pt 7 pi -1 ps 1.5
set style line 4 lt 7 lw 2 pt 9 pi -1 ps 1.5
set style line 5 lt 8 lw 2 pt 7 pi -1 ps 1.5
set style line 6 lt 8 lw 2 pt 9 pi -1 ps 1.5
# set pointintervalbox 2

# plot filename1 using (atan2($3, $2 - 0.8369)*180/pi):19 w l lt 12 lw 1 notitle "RL"
# plot filename1 using (atan2($3, $2 - 0.8369)*180/pi):(sqrt(($8-0.8369)**2 + $9**2 + $10**2)) w l lt 12 lw 1 notitle "RL"
plot filename5 using 1:($4*100) w linespoints ls 5 title "MLP, 10"
replot filename6 using 1:($4*100) w linespoints ls 6 title "RNN, 10"
replot filename3 using 1:($4*100) w linespoints ls 3 title "MLP, 5"
replot filename4 using 1:($4*100) w linespoints ls 4 title "RNN, 5"
replot filename1 using 1:($4*100) w linespoints ls 1 title "MLP, 1"
replot filename2 using 1:($4*100) w linespoints ls 2 title "RNN, 1"


set terminal epslatex standalone color colortext 12 lw 2 header \
"\\usepackage{amsmath}\n\\usepackage{bm}\n\\usepackage{siunitx}"
set output 'dist_r.tex'
replot