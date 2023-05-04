#Solve the station keeping RL problem
import argparse
import yaml
from pyrlprob.problem import RLProblem  #from pyrlprob.problem

# Input config file and parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="config_files/moon_station_keeping_config.yaml", \
    help='Input file with algorithm, model and environment config')
args = parser.parse_args()
config_file = args.config

#Define RL problem
SKproblem=RLProblem(config_file)

#Solve RL problem
trainer_dir, exp_dirs, last_cps, best_cp_dir = SKproblem.solve(evaluate=True, postprocess=True, debug=False)
