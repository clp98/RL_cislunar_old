from pyrlprob.problem import RLProblem


trainer_dir = 'results/PPO_2023-05-04_14-25-22/'
exp_dir = 'results/PPO_2023-05-04_14-25-22/PPO_2023-05-04_23-24-35/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_816ca_00000_0_2023-05-04_23-24-35/'
last_cp = 1955 

#Config file 
config_file = "config_files/moon_station_keeping_config.yaml" #exp_dir + "/config.yaml"

#Define RL problem
MSKProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = MSKProblem.evaluate(trainer_dir, exp_dir, last_cp)
