from pyrlprob.problem import RLProblem


# trainer_dir = 'results/BEAST/PPO_2023-05-08_06-46-18'
# exp_dir = '/home/carlo/RL_cislunar/results/BEAST/PPO_2023-05-08_06-46-18/'
# last_cp = 4000 

#Config file 
config_file = "/home/carlo/RL_cislunar/results/BEAST/PPO_2023-05-08_06-46-18/moon_station_keeping_config.yaml" #exp_dir + "/config.yaml"

#Define RL problem
MSKProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = MSKProblem.evaluate(trainer_dir, exp_dir, last_cp)

# #Config file 
# config_file = exp_dir + "/config.yaml"

# #Define RL problem
# UAVProblem = RLProblem(config_file)

# #Evaluation and postprocessing
# exp_dirs, last_cps, best_cp_dir = UAVProblem.evaluate(trainer_dir, 
#                                                       exp_dir, 
#                                                       last_cp)

