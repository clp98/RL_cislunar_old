filename1 = indir1.'theta_dist.txt'
filename2 = indir2.'theta_dist.txt'
filename3 = indir3.'theta_dist.txt'
filename4 = indir4.'theta_dist.txt'
filename5 = indir5.'theta_dist.txt'
filename6 = indir6.'theta_dist.txt'

set term unknown

set key bottom left nobox opaque height -1
set grid 

set xlabel "$\\beta,\\, \\si{deg}$"
set ylabel "$\\|\\bm{T}\\|,\\, \\si{N}$"
set xrange [-180:180]
set xtics -180, 60, 180
set yrange [0:0.12]

set format x "%.0f"
set format y "%.2f"

set style line 1 lt 6 lw 2 pt 7 pi -1 ps 1.5
set style line 2 lt 6 lw 2 pt 9 pi -1 ps 1.5
set style line 3 lt 7 lw 2 pt 7 pi -1 ps 1.5
set style line 4 lt 7 lw 2 pt 9 pi -1 ps 1.5
set style line 5 lt 8 lw 2 pt 7 pi -1 ps 1.5
set style line 6 lt 8 lw 2 pt 9 pi -1 ps 1.5
# set pointintervalbox 2

# plot filename1 using (atan2($3, $2 - 0.8369)*180/pi):19 w l lt 12 lw 1 notitle "RL"
# plot filename1 using (atan2($3, $2 - 0.8369)*180/pi):(sqrt(($8-0.8369)**2 + $9**2 + $10**2)) w l lt 12 lw 1 notitle "RL"

thrust_max = 0.109
factor = 2.7306165

set arrow from -180,thrust_max to 180,thrust_max nohead lt 2 dt 2 lw 3 front

plot filename5 using 1:($6*factor) w linespoints ls 5 title "MLP, 10"
replot filename6 using 1:($6*factor) w linespoints ls 6 title "RNN, 10"
replot filename3 using 1:($6*factor) w linespoints ls 3 title "MLP, 5"
replot filename4 using 1:($6*factor) w linespoints ls 4 title "RNN, 5"
replot filename2 using 1:($6*factor) w linespoints ls 2 title "RNN, 1"
replot filename1 using 1:($6*factor) w linespoints ls 1 title "MLP, 1"

set terminal epslatex standalone color colortext 14 lw 2 header \
"\\usepackage{amsmath}\n\\usepackage{bm}\n\\usepackage{siunitx}"
set output 'thrust.tex'
replot