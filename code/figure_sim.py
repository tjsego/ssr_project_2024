import argparse
import json
from matplotlib import pyplot as plt
from matplotlib import gridspec
import multiprocessing as mp
import os
from typing import List

import sim_lib
from sim_2_curator_py import CuratorAnalysis
from sim_2_modeler_py import SimulationReport

plt.rcParams['axes.labelpad'] = 2.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8.0
plt.rcParams['figure.constrained_layout.h_pad'] = 0.02
# plt.rcParams['figure.edgecolor'] = 'black'
plt.rcParams['figure.titlesize'] = 9.0
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.markersize'] = 1.0


def get_modeler_metadata(res_dir_modeler: str):
    with open(os.path.join(res_dir_modeler, 'sim_modeler.json'), 'r') as f:
        return sim_lib.Metadata.from_json(json.load(f))


def get_comparison(res_dir_curator: str):
    with open(os.path.join(res_dir_curator, 'sim_2_curator_pass.json'), 'r') as f:
        data = json.load(f)
    modeler_metadata = sim_lib.Metadata.from_json(data['modeler'])
    analysis = CuratorAnalysis.from_json(data['curator'])
    return modeler_metadata, analysis


def _get_comparison(res_dir_curator: str, conn):
    modeler_metadata, analysis = get_comparison(res_dir_curator)
    # conn.send(modeler_metadata, analysis)
    _find_greatest_error(modeler_metadata, analysis, conn)


def get_modeler_simdata(res_dir_modeler: str):
    with open(os.path.join(res_dir_modeler, 'simdata_modeler.json'), 'r') as f:
        return SimulationReport.from_json(json.load(f))


def _get_modeler_simdata(res_dir_modeler: str, conn):
    conn.send(get_modeler_simdata(res_dir_modeler))


def find_greatest_error(modeler_metadata: sim_lib.Metadata, analysis: CuratorAnalysis):

    err_max = -1.0
    res_oi_max = -1
    time_max = -1.0
    for name in analysis.var_names:
        for i in range(analysis.num_steps):
            eval_t = sim_lib.get_eval_info_times(*modeler_metadata.ecf_eval_info[name][i])
            ecf_curator = sim_lib.ecf(analysis.results[name][:modeler_metadata.sample_size, i], eval_t)
            err_i = sim_lib.ecf_compare(ecf_curator, modeler_metadata.ecf_evals[name][i])
            if err_i > err_max:
                time_max = modeler_metadata.sample_times[i]
                res_oi_max = int(i)
                err_max = err_i
    
    eval_t = {name: sim_lib.get_eval_info_times(*modeler_metadata.ecf_eval_info[name][res_oi_max]) for name in analysis.var_names}
    ecfs_modeler = {name: modeler_metadata.ecf_evals[name][res_oi_max] for name in analysis.var_names}
    ecfs_curator = {name: sim_lib.ecf(analysis.results[name][:modeler_metadata.sample_size, res_oi_max], eval_t[name]) 
                    for name in analysis.var_names}
    return time_max, eval_t, ecfs_modeler, ecfs_curator


def _find_greatest_error(modeler_metadata: sim_lib.Metadata, analysis: CuratorAnalysis, conn):
    conn.send(find_greatest_error(modeler_metadata, analysis))


def panel_trajectories(simdata: SimulationReport, sizes: List[int], axs):
    for i, size in enumerate(sizes):
        for j, name in enumerate(simdata.var_names):
            ax = axs[i][j]
            res = simdata.results[name]
            for k in range(size):
                ax.plot(simdata.results_times, res[k, :], alpha=0.01, color='gray')
            axs[i][j].ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(1, 5))

    for i, name in enumerate(simdata.var_names):
        axs[0][i].set_title(name).set_fontstyle('italic')


def panel_error_distributions(report: SimulationReport, 
                              size_sel: List[int], 
                              annotations: List[dict],
                              axs):
    for i, size_idx in enumerate(size_sel):
        axs[i].hist(report.ks_stats_samp_hist[size_idx], density=True)
    axs[0].set_title('EFECT error density').set_fontstyle('italic')

    for i, annot in enumerate(annotations):
        axs[i].text(transform=axs[i].transAxes, **annot)


def panel_error_hist(report: SimulationReport, ax):
    sizes = []
    means = []
    stdevs = []

    for sz, mn, sd in report.stat_hist:
        sizes.append(sz)
        means.append(mn)
        stdevs.append(sd)

    ax.errorbar(sizes, means, yerr=stdevs, color='black', marker='o', linestyle='none')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample size')
    ax.set_ylabel('EFECT error')


def panel_ecfs(eval_t, ecfs_modeler, ecfs_curator, var_names: List[str], xticks, xscale, axs):
    for i, name in enumerate(var_names):
        for j in range(2):
            axs[i][j].plot(eval_t[name] * xscale, ecfs_modeler[name][:, j], alpha=0.5)
            axs[i][j].plot(eval_t[name] * xscale, ecfs_curator[name][:, j], alpha=0.5)

    for i, name in enumerate(var_names):
        axs[i][0].set_ylabel(name).set_fontstyle('italic')
        axs[i][0].set_xticks(xticks[i])
        axs[i][1].set_xticks(xticks[i])
    axs[0][0].set_title('Real').set_fontstyle('italic')
    axs[0][1].set_title('Imaginary').set_fontstyle('italic')


def generate_figure(res_dir: str, output_dir: str = None, preview=False):

    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    print(f'Loading data:', res_dir)

    conn_comparison_1, conn_comparison_2 = mp.Pipe()
    p_comparison = mp.Process(target=_get_comparison, args=(res_dir, conn_comparison_1))
    p_comparison.start()

    modeler_simdata = get_modeler_simdata(os.path.join(res_dir, 'modeler_6'))
    # modeler_metadata, analysis = conn_comparison_2.recv()

    print('Generating plot...')

    fig = plt.figure(figsize=(4.72, 4.72), constrained_layout=True)
    gs0 = fig.add_gridspec(2, 1, height_ratios=(3, 4))

    label_kwargs = dict(
        fontsize=10,
        fontproperties={'weight': 'bold'}
    )
    subplot_kwargs = dict(
        wspace=0.001,
        hspace=0.001,
    )

    # Top: Modeler trajectories (left) and EFECT error distribution (right) with increasing sample size (top to bottom)
    size_sel = [0, 3, 4]
    gs_top = gridspec.GridSpecFromSubplotSpec(len(size_sel),
                                              len(modeler_simdata.var_names)+1, 
                                              subplot_spec=gs0[0],
                                              width_ratios=(1, 1, 1, 1, 3))
    sizes = [modeler_simdata.stat_hist[i][0] for i in size_sel]
    subfig_trajectories = fig.add_subfigure(gs_top[:, :len(modeler_simdata.var_names)])
    axs_trajectories = subfig_trajectories.subplots(3, 4, sharex=True, gridspec_kw=subplot_kwargs)
    panel_trajectories(modeler_simdata, sizes, axs_trajectories)
    subfig_trajectories.supxlabel('Time', fontsize=plt.rcParams['axes.labelsize'])
    subfig_trajectories.text(0.01, 0.99, 'A', ha='left', va='top', **label_kwargs)

    annotations = [
        dict(x=0.05 if sz < 1000 else 0.95,
             y=0.8,
             s=f'Sample size: {sz}', 
             va='top', 
             ha='left' if sz < 1000 else 'right',
             fontsize=7,
             bbox=dict(edgecolor='black', facecolor='white'))
        for sz in sizes
    ]

    subfig_error_distributions = fig.add_subfigure(gs_top[:, -1])
    axs_error_distributions = subfig_error_distributions.subplots(3, 1, sharex=True, gridspec_kw=subplot_kwargs)
    panel_error_distributions(modeler_simdata, size_sel, annotations, axs_error_distributions)
    subfig_error_distributions.supxlabel('EFECT error', fontsize=plt.rcParams['axes.labelsize'])
    subfig_error_distributions.text(0.01, 0.99, 'B', ha='left', va='top', **label_kwargs)

    # Bottom left: Modeler EFECT error vs. sample size
    gs_bot = gridspec.GridSpecFromSubplotSpec(len(modeler_simdata.var_names),
                                              3,
                                              subplot_spec=gs0[1],
                                              width_ratios=(4, 1, 1))
    subfig_error_hist = fig.add_subfigure(gs_bot[:, 0])
    ax_error_hist = subfig_error_hist.subplots(1, 1)
    panel_error_hist(modeler_simdata, ax_error_hist)
    subfig_error_hist.text(0.01, 0.99, 'C', ha='left', va='top', **label_kwargs)

    # Bottom right: Modeler vs. Curator ECFs
    xticks = [
        [0, 20],
        [0, 40],
        [0, 25],
        [0, 5]
    ]
    xscale = 1E5

    subfig_ecfs = fig.add_subfigure(gs_bot[:, 1:])
    axs_ecfs = subfig_ecfs.subplots(4, 2, sharey=True, gridspec_kw=subplot_kwargs)
    time_max, eval_t, ecfs_modeler, ecfs_curator = conn_comparison_2.recv()
    panel_ecfs(eval_t, ecfs_modeler, ecfs_curator, modeler_simdata.var_names, xticks, xscale, axs_ecfs)
    subfig_ecfs.supxlabel('Transform variable', fontsize=plt.rcParams['axes.labelsize'])
    subfig_ecfs.text(0.01, 0.99, 'D', ha='left', va='top', **label_kwargs)

    if preview:
        fig.show()
        plt.show(block=True)
    else:
        fig.savefig(os.path.join(output_dir, 'figure_workflowsim.png'))
        fig.savefig(os.path.join(output_dir, 'figure_workflowsim.svg'))


def main():
    pass


class ArgParse(argparse.ArgumentParser):
    
    def __init__(self):
        
        super().__init__()
        
        self.add_argument('-r', '--res-dir',
                          required=True,
                          type=str,
                          dest='res_dir',
                          help='Absolute path to the directory containing results')
        
        self.add_argument('-o', '--output-dir',
                          required=False,
                          type=str,
                          dest='output_dir',
                          default=os.path.dirname(__file__),
                          help='Output directory')

        self.add_argument('-p', '--preview',
                          required=False,
                          action='store_true',
                          help='Preview generated figure',
                          dest='preview')
        
        self.add_argument('-d', '--debug',
                          required=False,
                          action='store_true',
                          help='Run in debug mode (faster)',
                          dest='debug')
        
        self.parsed_args = self.parse_args()

    @property
    def res_dir(self):
        return self.parsed_args.res_dir

    @property
    def output_dir(self):
        return self.parsed_args.output_dir

    @property
    def preview(self):
        return self.parsed_args.preview

    @property
    def debug(self):
        return self.parsed_args.debug


if __name__ == '__main__':
    pa = ArgParse()
    generate_figure(res_dir=pa.res_dir, output_dir=pa.output_dir, preview=pa.preview)
