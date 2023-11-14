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
filename_out = "theta_dist.txt"

#Read file
L1 = np.array([0.8369, 0., 0.])
lconv = 384400. #km
vconv = 1.0245  #km/s
for exp in exp_dirs:

    file = exp + filename_in
    with open(file, "r") as f: # with open context
        f.readline()
        file_all = f.readlines()

        dist_r = []
        dist_v = []
        theta = []
        thrust = []
        r_Halo = []
        v_Halo = []

        dist_r_traj = []
        dist_v_traj = []
        theta_traj = []
        thrust_traj = []
        r_Halo_traj = []
        v_Halo_traj = []

        for line in file_all: #read line

            line_split = line.split()

            if len(line_split) > 0:
                data = np.array(line_split).astype(np.float64)

                dist_r_traj.append(data[18])
                dist_v_traj.append(data[19])
                theta_traj.append(np.arctan2(data[2], data[1]-L1[0])*180/np.pi)
                thrust_traj.append(data[17])
                r_Halo_traj.append(np.sqrt(data[7]**2 + data[8]**2 + data[9]**2)*lconv)
                v_Halo_traj.append(np.sqrt(data[10]**2 + data[11]**2 + data[12]**2)*vconv*1e3)
            else:
                dist_r.append(dist_r_traj)
                dist_v.append(dist_v_traj)
                theta.append(theta_traj)
                thrust.append(thrust_traj)
                r_Halo.append(r_Halo_traj)
                v_Halo.append(v_Halo_traj)

                dist_r_traj = []
                dist_v_traj = []
                theta_traj = []
                thrust_traj = []
                r_Halo_traj = []
                v_Halo_traj = []

    f.close() #close file

    # Erase the null vetors in the lists
    for i in range(len(theta)-1, -1, -1):
        if len(theta[i]) == 0:
            del theta[i]
            del dist_r[i]
            del dist_v[i]
            del thrust[i]
            del r_Halo[i]
            del v_Halo[i]

    #Sort theta vector and all the other vectors accordingly
    theta_sorted = []
    dist_r_sorted = []
    dist_v_sorted = []
    thrust_sorted = []
    r_Halo_sorted = []
    v_Halo_sorted = []

    theta_sorted_traj = []
    dist_r_sorted_traj = []
    dist_v_sorted_traj = []
    thrust_sorted_traj = []
    r_Halo_sorted_traj = []
    v_Halo_sorted_traj = []

    for i in range(len(theta)):

        for j in range(len(theta[i])):
            idx = bisect.bisect_left(theta_sorted_traj, theta[i][j])
            theta_sorted_traj.insert(idx, theta[i][j])
            dist_r_sorted_traj.insert(idx, dist_r[i][j])
            dist_v_sorted_traj.insert(idx, dist_v[i][j])
            thrust_sorted_traj.insert(idx, thrust[i][j])
            r_Halo_sorted_traj.insert(idx, r_Halo[i][j])
            v_Halo_sorted_traj.insert(idx, v_Halo[i][j])
        
        theta_sorted.append(theta_sorted_traj)
        dist_r_sorted.append(dist_r_sorted_traj)
        dist_v_sorted.append(dist_v_sorted_traj)
        thrust_sorted.append(thrust_sorted_traj)
        r_Halo_sorted.append(r_Halo_sorted_traj)
        v_Halo_sorted.append(v_Halo_sorted_traj)

        theta_sorted_traj = []
        dist_r_sorted_traj = []
        dist_v_sorted_traj = []
        thrust_sorted_traj = []
        r_Halo_sorted_traj = []
        v_Halo_sorted_traj = []
    
    # compute dist_r and dist_v relative to r_Halo and v_Halo
    dist_r_rel_sorted = []
    dist_v_rel_sorted = []
    dist_r_rel_sorted_traj = []
    dist_v_rel_sorted_traj = []

    for i in range(len(dist_r_sorted)):
        for j in range(len(dist_r_sorted[i])):
            dist_r_rel_sorted_traj.append(dist_r_sorted[i][j]/r_Halo_sorted[i][j])
            dist_v_rel_sorted_traj.append(dist_v_sorted[i][j]/v_Halo_sorted[i][j])

        dist_r_rel_sorted.append(dist_r_rel_sorted_traj)
        dist_v_rel_sorted.append(dist_v_rel_sorted_traj)

        dist_r_rel_sorted_traj = []
        dist_v_rel_sorted_traj = []


    # Compute the vector with the average values at every step among the trajectories
    theta_avg = []
    dist_r_avg = []
    dist_v_avg = []
    thrust_avg = []
    dist_r_rel_avg = []
    dist_v_rel_avg = []

    for i in range(len(theta_sorted[0])):
        theta_avg.append(0.)
        dist_r_avg.append(0.)
        dist_v_avg.append(0.)
        thrust_avg.append(0.)
        dist_r_rel_avg.append(0.)
        dist_v_rel_avg.append(0.)

        for j in range(len(theta_sorted)):
            theta_avg[i] += theta_sorted[j][i]
            dist_r_avg[i] += dist_r_sorted[j][i]
            dist_v_avg[i] += dist_v_sorted[j][i]
            thrust_avg[i] += thrust_sorted[j][i]
            dist_r_rel_avg[i] += dist_r_rel_sorted[j][i]
            dist_v_rel_avg[i] += dist_v_rel_sorted[j][i]

        theta_avg[i] /= len(theta_sorted)
        dist_r_avg[i] /= len(dist_r_sorted)
        dist_v_avg[i] /= len(dist_v_sorted)
        thrust_avg[i] /= len(thrust_sorted)
        dist_r_rel_avg[i] /= len(dist_r_rel_sorted)
        dist_v_rel_avg[i] /= len(dist_v_rel_sorted)


    # Print on a file the average values of theta, dist_r and dist_v in columns
    file_out = exp + filename_out
    with open(file_out, "w") as f2:
        f2.write("%20s %20s %20s %20s %20s %20s\n" % ("theta", "dist_r", "dist_v", "dist_r_rel", "dist_v_rel", "thrust"))
        for i in range(len(dist_r_avg)):
            f2.write("%20.7f %20.7f %20.7f %20.7f %20.7f %20.7f\n" % (theta_avg[i], dist_r_avg[i], dist_v_avg[i], dist_r_rel_avg[i], dist_v_rel_avg[i], thrust_avg[i]))
    f2.close() #close file
    