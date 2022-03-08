#!/bin/bash

#Synchronize the remote and local sol_lowthrust/ directories via rsync
rsync -a -P -e ssh spectral03@10.60.110.170:/home/spectral03/Documenti/Lorenzo/Python_codes/RL_cislunar/results/ /home/spectral02/Documenti/Python_codes/RL_cislunar/results/