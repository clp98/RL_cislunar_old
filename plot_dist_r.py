import matplotlib.pyplot as plt
import numpy as np

dist_r = []
time = []
 
#results/.../episode-step_data.txt
with open('/home/carlo/RL_cislunar/results/PPO_2023-05-13_05-56-55/PPO_2023-05-13_18-15-48/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_dc3ee_00000_0_2023-05-13_18-15-48/checkpoint_002000/episode_step_data.txt', \
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
plt.xlabel('time [*T_system]')
plt.ylabel('delta_r [km]')
plt.show()