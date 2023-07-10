from pyrlprob.problem import RLProblem


trainer_dir = 'results/MLP_5halos/'
exp_dir = 'results/MLP_5halos/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_48be4_00000_0_2023-07-07_10-18-26/'
last_cp = 5000 

#Config file 
config_file = exp_dir + "/config.yaml"

#Define RL problem
MSKProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = MSKProblem.evaluate(trainer_dir, exp_dir, last_cp, debug=True)

