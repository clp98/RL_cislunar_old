from pyrlprob.problem import RLProblem


trainer_dir = 'results/PPO_2023-05-31_09-10-41_Prova buona/'
exp_dir = 'results/PPO_2023-05-31_09-10-41_Prova buona/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_b0626_00000_0_2023-05-31_09-10-41'
last_cp = 5000 

#Config file 
config_file = exp_dir + "/config.yaml"

#Define RL problem
MSKProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = MSKProblem.evaluate(trainer_dir, exp_dir, last_cp)

