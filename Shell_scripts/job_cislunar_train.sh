#!/bin/bash


rm -r results/

i_min=2
i_max=2

for ((i = i_min ; i <= i_max ; i+=1)); do
    python main_solve.py --config "config_files/config"${i}".yaml"
done

# echo "Program metaRL_MRR completed." | mail -s "Python program ended" lorenzo.federici@uniroma1.it
