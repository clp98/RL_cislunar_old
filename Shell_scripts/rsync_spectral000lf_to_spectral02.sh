#!/bin/bash

#Synchronize the remote and local sol_saved/ directories via rsync
rsync -a -P -e ssh spectral000lf@10.60.110.125:/home/spectral000lf/Documenti/Python_codes/RL_cislunar/results/ /home/spectral02/Documenti/Python_codes/RL_cislunar/results/