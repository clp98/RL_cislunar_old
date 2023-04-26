from pyrlprob.problem import RLProblem


trainer_dir = 'results/3BP_pertIC/'
exp_dir = 'results/3BP_pertIC/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_83e6b_00000_0_2023-04-25_12-00-59/'
last_cp = 1500 

#Config file 
config_file = exp_dir + "/config.yaml"

#Define RL problem
UAVProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = UAVProblem.evaluate(trainer_dir, exp_dir, last_cp)
