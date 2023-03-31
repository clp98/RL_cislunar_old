import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np

from pyrlprob.utils import plot_metric


#Input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--trainer_dir', type=str, \
    help='Trainer directory')
parser.add_argument('--exp_dirs', nargs='+', type=str, \
    help='Experiment directories')
parser.add_argument('--last_cps', nargs='+', type=int, \
    help='Last checkpoints of the experiments')
parser.add_argument('--best_cp_dir', type=str, \
    help='Best checkpoint directory')
args = parser.parse_args()
trainer_dir = args.trainer_dir
exp_dirs = args.exp_dirs
last_cps = args.last_cps
best_cp_dir = args.best_cp_dir

#Realize graphs of training metrics
metrics = ["episode_reward", "custom_metrics/c_viol_r", "custom_metrics/c_viol_v", "custom_metrics/c_viol_s", "custom_metrics/mf"]
labels = ["$G$", "$\\Delta {r}_f$", "$\\Delta {v}_f$", "$\\Delta {x}_f$", "$m$"]

plt.style.use("seaborn")
matplotlib.rc('font', size=22)
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble='\\usepackage{amsmath} \\usepackage{bm}')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

for i, metric in enumerate(metrics):
    fig = plot_metric(metric,
                      exp_dirs,
                      last_cps)
    ax = fig.gca()
    ax.tick_params(labelsize=22)
    if "episode_reward" in metric:
        plt.yscale('symlog', subsy=[2,3,4,5,6,7,8,9], linthreshy=0.005, linscaley=2.0)
        plt.xlim(0, last_cps[-1])
        plt.ylim(-1e1, -1e-2)
        plt.xticks(np.arange(0, last_cps[-1]+1, step=last_cps[-1]/5))
        ax.yaxis.grid(True, which='minor', linewidth=0.5, linestyle='dashed')
        ax.yaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')
        ax.xaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')
    elif "c_viol" in metric:
        plt.yscale('log')
        plt.xlim(0, last_cps[-1])
        plt.ylim(1e-4, 1e1)
        plt.xticks(np.arange(0, last_cps[-1]+1, step=last_cps[-1]/5))
        ax.yaxis.grid(True, which='minor', linewidth=0.5, linestyle='dashed')
        ax.yaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')
        ax.xaxis.grid(True, which='major', linewidth=0.5, linestyle='dashed')
    plt.xlabel('Training iteration, $k$', fontsize=22)
    plt.ylabel(labels[i], fontsize=22)
    ax.legend(fontsize=20, loc = 'best')
    fig.savefig(trainer_dir + metric.rpartition('/')[-1] + ".pdf", dpi=300)

#Realize trajectory and control graphs
os.system("./Shell_scripts/plot_traj.sh " + best_cp_dir)
os.system("./Shell_scripts/plot_ctrl.sh " + best_cp_dir)