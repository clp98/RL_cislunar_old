
filename3 = indir.'episode_step_data.txt'
# filename4 = 'PO_files/GPOPS_traj_equalC_43.txt'
filename4 = 'PO_files/GPOPS_traj_equalC_optimal.txt'

set term unknown

#set key tmargin left horizontal maxrows 1 nobox noopaque #height 0.5 width 1
set key top center box opaque height 0.5 width 1.3
set grid 
set xrange [0:*]
# set yrange [0:0.15]
#set size ratio -1
#set xtics 0.5
#set ytics 0.5
#set view equal xy
#unset key
#set view equal xyz
#set view 0

set xlabel "$k$"
set ylabel "$u,\\, \\si{N}$"

set format x "%.0f"
set format y "%.3f"

# set autoscale fix

set lt 10 dt '--' lw 3.5 lc rgb "red"
set lt 11 lw 1.5 lc rgb "blue"
set ls 1 pt 2 ps 1 lc rgb "black"
set ls 2 pt 6 ps 1 lc rgb "black"
set ls 3 pt 3 ps 1 lc rgb "red"
set ls 4 pt 7 ps 1 lt 2


plot filename4 using ($1/0.15):(x=$9, y=$10, sqrt(x**2 + y**2)) w l lt 4 lw 2 title "GPOPS"
replot filename3 using ($1/0.15):8 w l lt 8 lw 2 title "PPO"

# min = (GPVAL_Y_MIN < GPVAL_X_MIN ? GPVAL_Y_MIN : GPVAL_X_MIN)
# max = (GPVAL_Y_MAX > GPVAL_X_MAX ? GPVAL_Y_MAX : GPVAL_X_MAX)

# set xrange [GPVAL_X_MIN - 0.1*abs(GPVAL_X_MIN):GPVAL_X_MAX + 0.1*abs(GPVAL_X_MAX)]
# set yrange [GPVAL_Y_MIN - 0.1*abs(GPVAL_Y_MIN):GPVAL_Y_MAX + 0.1*abs(GPVAL_Y_MAX)]
# set size ratio -1

set terminal epslatex standalone color colortext 8 lw 2 header \
"\\usepackage{amsmath}\n\\usepackage{siunitx}"
set output 'control.tex'
replot