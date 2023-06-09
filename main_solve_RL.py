#Solve the station keeping RL problem

from pyrlprob.problem import RLProblem 

#Configuration file
config_file='config_files/moon_station_keeping_config.yaml'

#Define RL problem
SKproblem = RLProblem(config_file)  #Station Keeping problem

#Solve RL problem
trainer_dir, exp_dirs, last_cps, best_cp_dir = SKproblem.solve(evaluate=True, postprocess=True, debug=False)
