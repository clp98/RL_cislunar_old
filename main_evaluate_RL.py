from pyrlprob.problem import RLProblem

#Config file 
config_file = "/home/carlo/RL_cislunar/results/BEAST/PPO_2023-05-08_06-46-18/moon_station_keeping_config.yaml" #exp_dir + "/config.yaml"

#Define RL problem
MSKProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = MSKProblem.evaluate(trainer_dir, exp_dir, last_cp)
