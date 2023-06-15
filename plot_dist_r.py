import matplotlib.pyplot as plt
import numpy as np

dist_r = []
time = []
 
#results/.../episode-step_data.txt
with open('results/PPO_2023-06-14_14-14-38/PPO_2023-06-15_02-40-09/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_9e570_00000_0_2023-06-15_02-40-09/checkpoint_005928/episode_step_data.txt', \
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
plt.yscale('log')
plt.xlabel('time [T_system]')
plt.ylabel('delta_r [km]')
plt.show()