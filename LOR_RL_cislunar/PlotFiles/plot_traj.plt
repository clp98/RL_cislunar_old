
filename1 = 'PO_files/L1_lyap_orbits.txt'
filename2 = 'PO_files/L2_lyap_orbits.txt'
filename3 = indir.'episode_step_data.txt'
# filename4 = 'PO_files/GPOPS_traj_equalC_43.txt'
#filename4 = 'PO_files/GPOPS_traj_diffC_optimal.txt'
filename5 = indir.'episode_end_data.txt'

set term unknown

#set key tmargin left horizontal maxrows 1 nobox noopaque #height 0.5 width 1
set key top center box opaque height 0.5 width 1.3
set grid 
set xrange [0.78:1.22]
set yrange [-0.15:0.15]
set size ratio -1
#set xtics 0.5
#set ytics 0.5
#set view equal xy
#unset key
#set view equal xyz
#set view 0

set xlabel "$x$"
set ylabel "$y$"

set format x "%.2f"
set format y "%.2f"

set autoscale fix

set lt 10 dt '--' lw 3.5 lc rgb "red"
set lt 11 lw 1.5 lc rgb "blue"
set ls 1 pt 2 ps 1 lc rgb "black"
set ls 2 pt 6 ps 1 lc rgb "black"
set ls 3 pt 3 ps 1 lc rgb "red"
set ls 4 pt 7 ps 1 lt 2
set ls 5 pt 7 ps 1 lt 1


factor = 1.2
# dt = 0.1499948
mu = 0.01215059
x_Moon = 0.987849
x_L1 = 0.8369
x_L2 = 1.1557

#set object 1 circle front at rNN0,rNN1 size eps fillstyle border lt -1 dt '--' lw 1 

plot "<echo '0.8369 0'" with p ls 1 notitle
replot "<echo '0.8369 0'" using 1:2:(sprintf("%2s", "L1")) with labels offset char 0,0.9 notitle
replot "<echo '1.1557 0'" with p ls 1 notitle
replot "<echo '1.1557 0'" using 1:2:(sprintf("%2s", "L2")) with labels offset char 0,0.9 notitle
replot "<echo '0.987849 0'" with p ls 2 notitle
replot "<echo '0.987849 0'" using 1:2:(sprintf("%4s", "Moon")) with labels offset char 0,0.9 notitle
#replot filename4 using 2:3:($9*factor):($10*factor) with vectors head lc rgb "#FFCC99" lw 1 notitle #FF9933
replot filename3 using 2:3:($9*factor):($10*factor) with vectors head lc rgb "#A0A0A0" lw 1 notitle #FF9933
#replot filename3 using 2:3:($5*$10/dt*factor):($6*$10/dt*factor) with vectors filled head lc 6 lw 1 notitle "$T_{\\pi^{unp}}$"
replot for [k = 13:13] filename1 i k using 1:2 w l lt 6 lw 2 notitle "L1 Lyapunov orbit" #0:40:1
replot for [k = 21:21] filename2 i k using 1:2 w l lt 7 lw 2 notitle "L2 Lyapunov orbit" #0:40:1 #18 #21
# replot filename4 using 2:3 w l lt 4 lw 2 title "GPOPS"
replot filename3 using 2:3 w l lt 8 lw 2 title "PPO"
# replot "<echo '".rNN0." ".rNN1."'" with p ls 3 notitle
# replot "<echo '".rCP0." ".rCP1."'" with p ls 4 notitle
# replot filename4 using 2:3 every ::1134::1134 w p ls 5 notitle "GPOPS"

# min = (GPVAL_Y_MIN < GPVAL_X_MIN ? GPVAL_Y_MIN : GPVAL_X_MIN)
# max = (GPVAL_Y_MAX > GPVAL_X_MAX ? GPVAL_Y_MAX : GPVAL_X_MAX)

# set xrange [GPVAL_X_MIN - 0.1*abs(GPVAL_X_MIN):GPVAL_X_MAX + 0.1*abs(GPVAL_X_MAX)]
# set yrange [GPVAL_Y_MIN - 0.1*abs(GPVAL_Y_MIN):GPVAL_Y_MAX + 0.1*abs(GPVAL_Y_MAX)]
# set size ratio -1

set terminal epslatex standalone color colortext 8 lw 2 header \
"\\usepackage{amsmath}\n\\usepackage{siunitx}"
set output 'trajectory.tex'
replot