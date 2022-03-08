import argparse
import os

from pyrlprob.problem import RLProblem


#Input config file and parameters
parser = argparse.ArgumentParser()
parser.add_argument('--trainer_dir', type=str, \
    help='Trainer directory')
parser.add_argument('--exp_dir', type=str, \
    help='Experiment directory')
parser.add_argument('--last_cp', type=int, \
    help='Last checkpoint')
parser.add_argument('--graphs', type=bool, default=True, \
    help='Whether to realize graphs')
args = parser.parse_args()
trainer_dir = args.trainer_dir
exp_dir = args.exp_dir
last_cp = args.last_cp
graphs = args.graphs

#Config file
config_file = exp_dir + "/config.yaml"

#Define RL problem
MRRProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = MRRProblem.evaluate(trainer_dir, exp_dir, last_cp, debug=False)

#Graphs
exp_dirs_str = ""
last_cps_str = ""
for i in range(len(exp_dirs)):
    exp_dirs_str += " " + exp_dirs[i]
    last_cps_str += " " + str(last_cps[i])
if graphs:
    os.system('python main_graphs.py ' + \
                ' --trainer_dir ' + trainer_dir + \
                ' --exp_dirs' + exp_dirs_str + \
                ' --last_cps' + last_cps_str + \
                ' --best_cp_dir ' + best_cp_dir)
