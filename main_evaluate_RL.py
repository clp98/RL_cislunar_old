from pyrlprob.problem import RLProblem


trainer_dir = 'results/PPO_2023-04-28_13-13-47/'
exp_dir = 'results/PPO_2023-04-28_13-13-47/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_2eed6_00000_0_2023-04-28_13-13-47/'
last_cp = 400 

#Config file 
config_file = "config_files/moon_station_keeping_config.yaml" #exp_dir + "/config.yaml"

#Define RL problem
UAVProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = UAVProblem.evaluate(trainer_dir, exp_dir, last_cp)
