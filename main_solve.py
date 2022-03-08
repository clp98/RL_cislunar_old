import argparse
import os

from pyrlprob.problem import RLProblem


#Input config file and parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="config_files/config0.yaml", \
    help='Input file with algorithm, model and environment config')
parser.add_argument('--postproc', type=bool, default=True, \
    help='Whether to do evaluation and postprocessing')
parser.add_argument('--graphs', type=bool, default=True, \
    help='Whether to realize graphs')
args = parser.parse_args()
config_file = args.config
postproc = args.postproc
graphs = args.graphs

#Define RL problem
MRRProblem = RLProblem(config_file)

#Solve RL problem
trainer_dir, exp_dirs, last_cps, best_cp_dir = \
        MRRProblem.solve(evaluate=postproc, 
                         postprocess=postproc,
                         debug=False)

#Graphs
exp_dirs_str = ""
last_cps_str = ""
for i in range(len(exp_dirs)):
    exp_dirs_str += " " + exp_dirs[i]
    last_cps_str += " " + str(last_cps[i])
if postproc and graphs:
    os.system('python main_graphs.py ' + \
                ' --trainer_dir ' + trainer_dir + \
                ' --exp_dirs' + exp_dirs_str + \
                ' --last_cps' + last_cps_str + \
                ' --best_cp_dir ' + best_cp_dir)
