import argparse
from ast import For
import os
import platform
from sympy import divisors
import yaml
import ray

from pyrlprob.problem import RLProblem


# Input config file and parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="config_files/moon_station_keeping_config.yaml", \
    help='Input file with algorithm, model and environment config')
args = parser.parse_args()
config_file = args.config
config = yaml.safe_load(open(config_file))

# Tests
hardware = ["cpu_only"] #"gpu_d_cpu_w"] #, "gpu_w_cpu_d", "gpu_only"
cpus = 8.0
gpus = 0
envs = 100
max_w = 20

# Output file
f_log = open("./cpu_times.txt", "w")
f_log.write("%20s %20s %20s %20s %20s %20s %20s %20s\n" \
    % ("# hardware", "workers", "envs_per_worker", \
    "cpus_per_w", "gpus_per_w", "cpus_per_d", "gpus_per_d", "time[s]"))

ray.init(logging_level="ERROR", log_to_driver=False)

# Run simulation
for h in hardware:
    for w in divisors(envs):
        
        if w > max_w:
            continue
        
        eval_w = w
        total_w = w + eval_w

        if h == "cpu_only":
            cpus_per_w = (cpus - 1.)/total_w if (cpus - 1.)/total_w < 1 else int((cpus - 1.)/total_w)
            cpus_per_d = int(cpus - cpus_per_w*total_w)
            gpus_per_w = 0
            gpus_per_d = 0
        elif h == "gpu_only":
            cpus_per_w = 0
            cpus_per_d = 0
            gpus_per_w = gpus/(total_w + 1)
            gpus_per_d = gpus_per_w
        elif h == "gpu_d_cpu_w":
            cpus_per_w = cpus/total_w if cpus/total_w < 1 else int(cpus/total_w)
            cpus_per_d = int(cpus - cpus_per_w*total_w)
            gpus_per_w = 0
            gpus_per_d = gpus
        elif h == "gpu_w_cpu_d":
            cpus_per_w = 0
            cpus_per_d = cpus
            gpus_per_w = gpus/total_w
            gpus_per_d = 0
        
        # Training config
        config["stop"]["training_iteration"] = 1
        config["config"]["num_workers"] = w
        config["config"]["num_envs_per_worker"] = int(envs / w)
        config["config"]["num_cpus_per_worker"] = cpus_per_w
        config["config"]["num_cpus_for_driver"] = cpus_per_d
        config["config"]["num_gpus_per_worker"] = gpus_per_w
        config["config"]["num_gpus"] = gpus_per_d
        config["config"]["create_env_on_driver"] = False

        #Evaluation config
        config["config"]["evaluation_parallel_to_training"] = True
        config["config"]["evaluation_interval"] = 1
        config["config"]["evaluation_num_episodes"] = eval_w
        config["config"]["evaluation_num_workers"] = eval_w
        config["config"]["evaluation_config"]["explore"] = False

        # Define RL problem
        Prb = RLProblem(config)

        # Solve RL problem
        trainer_dir, exp_dirs, last_cps, best_cp_dir, run_time = \
                Prb.solve(evaluate=False, postprocess=False, debug=False, 
                    open_ray=False, return_time=True)
        
        # Print results
        f_log.write("%20s %20d %20d %20.5f %20.5f %20.5f %20.5f %20.5f\n" \
            % (h, w, int(envs / w), cpus_per_w, gpus_per_w, cpus_per_d, gpus_per_d, run_time))
        
        print("Done case: w = %d, cpu_per_w = %4.3f" % (w, cpus_per_w))

f_log.close()
ray.shutdown()