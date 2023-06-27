from pyrlprob.problem import RLProblem


trainer_dir = 'results/PPO_2023-06-26_13-56-09/'
exp_dir = 'results/PPO_2023-06-26_13-56-09/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_e09a0_00000_0_2023-06-26_13-56-09/'
last_cp = 6000 

#Config file 
config_file = exp_dir + "/config.yaml"

#Define RL problem
MSKProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = MSKProblem.evaluate(trainer_dir, exp_dir, last_cp)

