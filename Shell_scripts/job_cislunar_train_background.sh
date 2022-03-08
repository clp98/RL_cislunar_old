#!/bin/bash


mkdir -p StdOutput/
nohup ./Shell_scripts/job_cislunar_train.sh > StdOutput/RL_cislunar.out 2> StdOutput/RL_cislunar.err &