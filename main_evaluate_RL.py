from pyrlprob.problem import RLProblem


trainer_dir = 'results/PPO_2023-05-01_12-25-48/'
exp_dir = 'results/PPO_2023-05-01_12-25-48/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_fa4c0_00000_0_2023-05-01_12-25-48/'
last_cp = 659 

#Config file 
config_file = "config_files/moon_station_keeping_config.yaml" #exp_dir + "/config.yaml"

#Define RL problem
UAVProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = UAVProblem.evaluate(trainer_dir, exp_dir, last_cp)
