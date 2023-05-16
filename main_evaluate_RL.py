from pyrlprob.problem import RLProblem


trainer_dir = 'results/TOP/'
exp_dir = '/home/jupiter/Documents/carlo/RL_cislunar/results/TOP/'
last_cp = 2000 

#Config file 
config_file = "config_files/moon_station_keeping_config.yaml" #exp_dir + "/config.yaml"

#Define RL problem
MSKProblem = RLProblem(config_file)

#Evaluation and postprocessing
exp_dirs, last_cps, best_cp_dir = MSKProblem.evaluate(trainer_dir, exp_dir, last_cp)

#print(self.state['r']-self.r_Halo) #riesci a girare solo la valutazione con queste nuove quantità senza riaddestrare la rete?