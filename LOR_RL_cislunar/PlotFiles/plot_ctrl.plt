
filename1 = indir.'episode_step_data.txt'
filename2 = indir.'episode_end_data.txt'

set term unknown

#set key tmargin left horizontal maxrows 1 nobox noopaque #height 0.5 width 1
set key top right nobox opaque #height 0.5 width 1.3
set grid 
# set size ratio -1
#set xtics 0.1
set ytics 0.02
#set view equal xy
unset key
#set view equal xyz
#set view 0

set xlabel "$t$"
set ylabel "$u$"

set format x "%.2f"
set format y "%.2f"

#set autoscale fix

set lt 10 dt '--' lw 2 lc rgb "black"
set lt 11 lw 1.5 lc rgb "blue"
set lt 12 lc rgb "#A0A0A0"
set ls 1 pt 2 ps 1 lc rgb "black"
set ls 2 pt 6 ps 1 lc rgb "black"
set ls 3 pt 3 ps 1 lc rgb "red"
set ls 4 pt 7 ps 0.4 lc rgb "black"
set ls 5 pt 5 ps 0.6 lc rgb "black"

stats filename2 using 18 name "tf"

set arrow 1 from 0,0.1 to tf_max,0.1 nohead lt 10 front

plot filename1 using 1:12 w l lt 12 lw 0.7 notitle "RL"

# set yrange [0:0.11]
set xrange [0:tf_max]

set terminal epslatex standalone color colortext 12 lw 2 header \
"\\usepackage{amsmath}\n\\usepackage{siunitx}"
set output 'control.tex'
replot