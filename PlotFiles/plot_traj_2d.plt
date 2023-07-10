
#filename1 = 'Halo_L1_rv.txt'
# filename2 = 'PO_files/L2_halo_north_orbits.txt'
filename2 = indir.'episode_step_data.txt'

set term unknown

#set key tmargin left horizontal maxrows 1 nobox noopaque #height 0.5 width 1
set key top center nobox height 1
set key spacing 1.4
# set xrange [0.78:1.22]
# set yrange [-0.15:0.15]
set size ratio -1
set xtics offset 0.1,0
set ytics offset -1.5,0
#set view equal xy
# unset key
# set view equal xyz
# set view 115,60
# set grid

set xlabel "$x$"
set ylabel "$y$"
# set zlabel "$z$"

set format x "%.2f"
set format y "%.2f"
# set format z "%.2f"

set autoscale fix

set lt 10 dt '--' lw 3.5 lc rgb "red"
set lt 11 lw 1.5 lc rgb "blue"
set ls 1 pt 2 ps 1 lc rgb "black"
set ls 2 pt 6 ps 1 lc rgb "black"


# factor = 3
# dt = 0.1499948
mu = 0.01215059

x_Moon = 0.987849
x_L1 = 0.8369
x_L2 = 1.1557


plot "<echo '0.8369 0'" with p ls 1 notitle
replot "<echo '0.8369 0'" using 1:2:3:(sprintf("%2s", "L1")) with labels offset char 0,0.7 notitle
replot filename2 using 7:8 w l lt 6 lw 2.5 notitle
# replot filename2 using 1:2 w l lt 7 lw 2.5 notitle 

min = (GPVAL_Y_MIN < GPVAL_X_MIN ? GPVAL_Y_MIN : GPVAL_X_MIN)
max = (GPVAL_Y_MAX > GPVAL_X_MAX ? GPVAL_Y_MAX : GPVAL_X_MAX)

set xrange [GPVAL_X_MIN - 0.1*abs(GPVAL_X_MIN):GPVAL_X_MAX + 0.1*abs(GPVAL_X_MAX)]
set yrange [GPVAL_Y_MIN - 0.1*abs(GPVAL_Y_MIN):GPVAL_Y_MAX + 0.1*abs(GPVAL_Y_MAX)]
set size ratio -1

set terminal epslatex standalone color colortext 10 lw 2 header \
"\\usepackage{amsmath}\n\\usepackage{siunitx}"
set output 'trajectory_2d.tex'
replot