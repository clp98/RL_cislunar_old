
filename1 = 'PO_files/L1_halo_north_orbits.txt'
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
set view 115,60
# set grid

set xlabel "$x$"
set ylabel "$y$"
set zlabel "$z$"

set format x "%.2f"
set format y "%.2f"
set format z "%.2f"

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

# splot "<echo '0.8369 0 0'" with p ls 1 notitle
# replot "<echo '0.8369 0 0'" using 1:2:(sprintf("%2s", "L1")) with labels offset char 0,0.9 notitle
splot "<echo '1.1557 0 0'" with p ls 1 notitle
replot "<echo '1.1557 0 0'" using 1:2:3:(sprintf("%2s", "L2")) with labels offset char 0,0.7 notitle
# replot "<echo '0.987849 0 0'" with p ls 2 notitle
# replot "<echo '0.987849 0 0'" using 1:2:3:(sprintf("%4s", "Moon")) with labels offset char 0,0.9 notitle
#replot filename4 using 2:3:($9*factor):($10*factor) with vectors nohead lc rgb "#fed8b1" lw 1 notitle "$T_{\\text{ind}}$" #"#FFA500"
#replot filename3 using 2:3:($5*$10/dt*factor):($6*$10/dt*factor) with vectors filled head lc 6 lw 1 notitle "$T_{\\pi^{unp}}$"
replot for [k = 1] filename1 i k using 1:2:3 w l lt 6 lw 2.5 notitle "$Ly_1$" #0:40:1
# replot for [k = 21:21] filename2 i k using 1:2 w l lt 7 lw 2.5 title "$Ly_2^{(A)}$" #0:40:1
# replot for [k = 18:18] filename2 i k using 1:2 w l lt 4 lw 2.5 title "$Ly_2^{(B)}$" #0:40:1
replot for [k = 1] filename2 using 1:2 w l lt 7 lw 2 notitle 
# replot filename2 every ::::0 using 1:2 w p ls 3 notitle "Arrival"
# replot filename2 every ::::0 using 1:2:(sprintf("%5s", "Earth")) with labels offset char -3.2,0 notitle #font "Times, 20" notitle "Target ID"
# replot filename2 every ::400 using 1:2:(sprintf("%4s", "Mars")) with labels offset char -1.7,0.9 notitle

# min = (GPVAL_Y_MIN < GPVAL_X_MIN ? GPVAL_Y_MIN : GPVAL_X_MIN)
# max = (GPVAL_Y_MAX > GPVAL_X_MAX ? GPVAL_Y_MAX : GPVAL_X_MAX)

# set xrange [GPVAL_X_MIN - 0.1*abs(GPVAL_X_MIN):GPVAL_X_MAX + 0.1*abs(GPVAL_X_MAX)]
# set yrange [GPVAL_Y_MIN - 0.1*abs(GPVAL_Y_MIN):GPVAL_Y_MAX + 0.1*abs(GPVAL_Y_MAX)]
# set size ratio -1

set terminal epslatex standalone color colortext 10 lw 2 header \
"\\usepackage{amsmath}\n\\usepackage{siunitx}"
set output 'trajectory.tex'
replot