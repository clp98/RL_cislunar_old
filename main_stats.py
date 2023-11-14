import numpy as np
from numpy.linalg import norm
import bisect

#Solutions
exp_dirs = [
    "results/MLP_1halo/PPO_2023-08-04_09-08-29/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_26e19_00000_0_2023-08-04_09-08-29/checkpoint_004886/",
    "results/LSTM_1halo/PPO_2023-08-07_11-03-36/PPO_2023-08-07_12-02-11/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_e9ef2_00000_0_2023-08-07_12-02-11/checkpoint_004981/",
    "results/MLP_5halos/PPO_2023-08-03_13-25-53/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_f1a37_00000_0_2023-08-03_13-25-53/checkpoint_004732/",
    "results/LSTM_5halos/PPO_2023-08-03_14-27-55/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_9c7b0_00000_0_2023-08-03_14-27-56/checkpoint_004718/",
    "results/MLP_10halos/PPO_2023-08-03_14-38-34/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_190ec_00000_0_2023-08-03_14-38-34/checkpoint_004928/",
    "results/LSTM_10halos/PPO_2023-08-03_15-14-19/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_17934_00000_0_2023-08-03_15-14-19/checkpoint_004810/"
    ]

#Input and output file
filename_in = "episode_step_data.txt"
filename_out = "success_rates.txt"

#Read file
eps_r = 20.
eps_v = 1.
m0 = 1000.
for exp in exp_dirs:

    dist_r = []
    dist_v = []
    dist_x = []
    mf = []

    file = exp + filename_in
    with open(file, "r") as f: # with open context
        f.readline()
        file_all = f.readlines()

        dist_r_traj = []
        dist_v_traj = []
        # dist_x_traj = []
        m_traj = []

        for line in file_all: #read line

            if line.strip() == "":

                if len(dist_r_traj):
                    dist_r.append(sum(dist_r_traj)/len(dist_r_traj))
                    dist_v.append(sum(dist_v_traj)/len(dist_v_traj))
                    dist_x.append(max(dist_r[-1]-eps_r, dist_v[-1]-eps_v))
                    mf.append(m_traj[-1])

                dist_r_traj = []
                dist_v_traj = []
                m_traj = []
            else:

                line = line.split()
                data = np.array(line).astype(np.float64)

                dist_r_traj.append(data[18])
                dist_v_traj.append(data[19])
                # dist_x_traj.append(max(data[18]-eps_r, data[19]-eps_v))
                m_traj.append(data[13])
    f.close() #close file

    n_traj = len(mf)
    
    #Sort data
    dist_r.sort()
    dist_v.sort()
    dist_x.sort()
    mf.sort(reverse=True)

    #Find SR
    sr_x = ((bisect.bisect_left(dist_x, 0.)))/n_traj*100.

    #Find sigma levels
    sigmas = [0.683*n_traj, 0.955*n_traj, 0.997*n_traj]
    # sigmas = [0.5*n_traj, 0.7*n_traj, 0.9*n_traj]
    sigmas_r = [dist_r[int(sigma) - 1] for sigma in sigmas]
    sigmas_v = [dist_v[int(sigma) - 1] for sigma in sigmas]
    sigmas_m = [mf[int(sigma) - 1] for sigma in sigmas]

    # #Print results
    f_out = open(exp + filename_out, "w")
    f_out.write("%30s %10.7f\n" % ("success rate:", sr_x))
    f_out.write("%30s %10.7f %10.7f %10.7f\n" % ("sigmas on position [km]:", sigmas_r[0], sigmas_r[1], sigmas_r[2]))
    f_out.write("%30s %10.7f %10.7f %10.7f\n" % ("sigmas on velocity [m/s]:", sigmas_v[0], sigmas_v[1], sigmas_v[2]))
    f_out.write("%30s %10.7f %10.7f %10.7f\n" % ("sigmas on prop mass [kg]:", m0*(1 - sigmas_m[0]), m0*(1 - sigmas_m[1]), m0*(1 - sigmas_m[2])))
    f_out.close()