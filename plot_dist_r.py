import matplotlib.pyplot as plt
import numpy as np


dist_r = []
time = []

with open('results/PPO_2023-05-08_22-15-29/PPO_2023-05-09_08-17-38/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_a266a_00000_0_2023-05-09_08-17-38/checkpoint_001942/episode_step_data.txt', \
          'r') as file_stepdata:
    file_stepdata.readline()
    file = file_stepdata.readlines()
    for line in file:
        line=line.split()
        line_array=np.array(line).astype(np.float64)

        if len(line_array)>=1:
            dist_r.append(line_array[12])
            time.append(line_array[7])

file_stepdata.close()

plt.plot(time, dist_r)
plt.xlabel('time')
plt.ylabel('dist_r')
plt.show()