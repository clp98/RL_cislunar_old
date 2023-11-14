import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pyrlprob.utils import plot_metric

#Plot style
plt.style.use("seaborn")  #funziona??
matplotlib.rc('font', size=18)
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble='\\usepackage{amsmath} \\usepackage{bm} \\usepackage{siunitx}')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

#Solutions
out_dir = "results/compare/"
exp_dirs = [["results/MLP_1halo/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_2be69_00000_0_2023-08-03_17-52-22/"], \
    ["results/LSTM_1halo/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_babf9_00000_0_2023-08-04_09-12-37/",
     "results/LSTM_1halo/PPO_2023-08-04_15-17-38/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_b8908_00000_0_2023-08-04_15-17-38/",
     "results/LSTM_1halo/PPO_2023-08-07_11-03-36/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_bb0fd_00000_0_2023-08-07_11-03-36/"], \
    ["results/MLP_5halos/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_4640d_00000_0_2023-07-11_11-39-31/", 
     "results/MLP_5halos/PPO_2023-07-10_14-50-24/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_c64e6_00000_0_2023-07-10_14-50-24/"], \
    ["results/LSTM_5halos/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_26efb_00000_0_2023-07-11_14-23-17/"],
    ["results/MLP_10halos/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_9231e_00000_0_2023-07-12_12-23-24/"],
    ["results/LSTM_10halos/PPO_environment.Moon_Station_Keeping_Env.Moon_Station_Keeping_Env_51a68_00000_0_2023-07-13_03-23-33/"]]
last_cps = [[5000], [1358, 4766, 5000], [620, 5000], [5000], [5000], [5000]]
max_cp = 5000
label_sol = ["$\\bm{\\pi}^{\\text{MLP}},\\, M=1$", "$\\bm{\\pi}^{\\text{RNN}},\\, M=1$", \
             "$\\bm{\\pi}^{\\text{MLP}},\\, M=5$", "$\\bm{\\pi}^{\\text{RNN}},\\, M=5$", \
             "$\\bm{\\pi}^{\\text{MLP}},\\, M=10$", "$\\bm{\\pi}^{\\text{RNN}},\\, M=10$"]
n_sol = len(last_cps)

#Metrics to plot
metrics = ["episode_reward", "custom_metrics/dist_r_mean", "custom_metrics/dist_v_mean", "custom_metrics/mf"]
labels = ["$G$", "$\\bar{d}_r,\\, \\si{km}$", "$\\bar{d}_v,\\, \\si{m/s}$", "$m_L,\\, \\si{kg}$"]

for i, metric in enumerate(metrics):

    #Create figure
    fig = plt.figure()
    fig.set_size_inches(10.4,6.4)
    
    #Set graph properties
    ax = fig.gca()
    ax.tick_params(labelsize=22)
    plt.xlim(0, max_cp)
    plt.xticks(np.arange(0,  max_cp+1, step= max_cp/5))
    ax.yaxis.grid(True, which='minor', linewidth=0.5, linestyle='dashed')
    ax.yaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')
    ax.xaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')

    #Plot metrics
    if "dist_r" in metric:
        fig = plot_metric("custom_metrics/epsilon",
                        exp_dirs[n_sol-1],
                        last_cps[n_sol-1],
                        window=4,
                        step=1,
                        factor=200.,
                        fig=fig,
                        label="$\\epsilon_r$",
                        color="black")
    elif "dist_v" in metric:
        fig = plot_metric("custom_metrics/epsilon",
                        exp_dirs[n_sol-1],
                        last_cps[n_sol-1],
                        window=4,
                        step=1,
                        factor=10,
                        fig=fig,
                        label="$\\epsilon_v$",
                        color="black")
        
    for j in range(n_sol):
        fig = plot_metric(metric,
                         exp_dirs[j],
                         last_cps[j],
                         window=4,
                         step=4,
                         factor=1.0,
                         fig=fig,
                         label=label_sol[j],
                         color=palette[j])
    
    if "episode_reward" in metric:
        plt.yscale('symlog', subs=[2,3,4,5,6,7,8,9], linthresh=5.0e-06, linscale=2.0)
        plt.ylim(-0.1, -5.0e-06)
    elif "dist_r" in metric:
        plt.yscale('log')
        #plt.ylim(0.03, 200)
    elif "dist_v" in metric:
        plt.yscale('log')
        #plt.ylim(0.005, 50)
        
    
    plt.xlabel('Training iteration, $n$', fontsize=22)
    plt.ylabel(labels[i], fontsize=22)

    #get handles and labels
    handles, leg_labels = plt.gca().get_legend_handles_labels()

    #specify order of items in legend
    if "dist" in metric:
        # order = [0,4,1,2,3]
        order = [1, 2, 3, 4, 5, 6]
    else:
        order = [0, 1, 2, 3, 4, 5]

    #add legend to plot
    ax.legend([handles[idx] for idx in order],[leg_labels[idx] for idx in order], fontsize=18, bbox_to_anchor=(0.5, 1.17), loc = 'upper center', ncol=3)

    #ax.legend(fontsize=15, loc = 'upper center', ncol=4)
    fig.savefig(out_dir + metric.rpartition('/')[-1] + ".pdf", dpi=300)