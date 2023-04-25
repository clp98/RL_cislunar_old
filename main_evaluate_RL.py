from pyrlprob.problem import RLProblem


trainer_dir = 'results/PPO_2023-04-21_11-18-18/'
exp_dir = 'results/PPO_2023-04-21_11-18-18/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_e4252_00000_0_2023-04-21_11-18-18/'
last_cp = 1500 

#Config file 
config_file = exp_dir + "/config.yaml"

#Define RL problem
UAVProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = UAVProblem.evaluate(trainer_dir, exp_dir, last_cp)
