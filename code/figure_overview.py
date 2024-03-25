import argparse
import json
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from typing import Dict, List, Tuple

try:
    from .stochastic_tests import Test
except ImportError:
    from stochastic_tests import Test

plt.rcParams['axes.labelpad'] = 2.0
plt.rcParams['axes.labelsize'] = 8.0
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8.0
plt.rcParams['figure.constrained_layout.h_pad'] = 0.02
# plt.rcParams['figure.edgecolor'] = 'black'
plt.rcParams['figure.labelsize'] = 8.0
plt.rcParams['figure.titlesize'] = 9.0
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['lines.markersize'] = 1.0

DEBUG = False


def load_test(fp):
    return Test.load(fp)


def _panel_coinfection(_test: Test, axs):
    trials_oi = _test.trials[-1] if not DEBUG else _test.trials[1]
    sims_s = _test.sims_s[trials_oi]

    org = [(0, 'T'), (1, 'I1'), (2, 'I2'), (3, 'V'), (4, 'P')]

    time_data = sims_s.results_time

    for i, name in org:
        res = sims_s.results[name]
        for k in range(res.shape[0]):
            axs[i].plot(time_data, res[k, :], alpha=0.01, color='gray')
        axs[i].set_xticks([0, 9, 18])
        axs[i].set_title(name).set_fontstyle('italic')


def panel_coinfection(_test: Test):
    fig, axs = plt.subplots(1, 5, figsize=(4.5, 1.0), layout='compressed')

    _panel_coinfection(_test, axs)

    return fig, axs


def _panel_lorentz(_test: Test, axs):
    trials_oi = _test.trials[-1] if not DEBUG else _test.trials[1]
    sims_s = _test.sims_s[trials_oi]

    org = [(0, 'x'), (1, 'y'), (2, 'z')]

    time_data = sims_s.results_time

    for i, name in org:
        res = sims_s.results[name]
        for k in range(res.shape[0]):
            axs[i].plot(time_data, res[k, :], alpha=0.01, color='gray')
        axs[i].set_title(name).set_fontstyle('italic')


def panel_lorentz(_test: Test):
    fig, axs = plt.subplots(1, 3, figsize=(4.5, 1.0), layout='compressed')

    _panel_lorentz(_test, axs)

    return fig, axs


def _panel_sir(_test: Test, axs):
    trials_oi = _test.trials[-1] if not DEBUG else _test.trials[1]
    sims_s = _test.sims_s[trials_oi]

    org = [(0, 0, 'S'),
           (0, 1, 'I'),
           (1, 0, 'R'),
           (1, 1, 'V')]

    time_data = sims_s.results_time

    for i, j, name in org:
        res = sims_s.results[name]
        for k in range(res.shape[0]):
            axs[i][j].plot(time_data, res[k, :], alpha=0.01, color='gray')
        axs[i][j].ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=(1, 5))
        axs[i][j].set_title(name).set_fontstyle('italic')


def panel_sir(_test: Test):
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(2, 2), layout='compressed')

    _panel_sir(_test, axs)

    return fig, axs


def _panel_error_metric(_test: Test, ax):
    _test.generate_ecf_sampling_fits()

    for i, data_f in enumerate(_test.ecf_sampling_fits[0]):
        if data_f is not None:
            ax.plot(_test.trials, [Test.ecf_diff_fit_func(n, *data_f) for n in _test.trials],
                    label=f'Sample size {_test.trials[i + 1]}')

    avg = np.asarray([_test.ecf_sampling[t][0] for t in _test.trials], dtype=float)
    std = np.asarray([_test.ecf_sampling[t][1] for t in _test.trials], dtype=float)
    ax.errorbar(_test.trials, avg, yerr=std, color='black', marker='o', linestyle='none')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sample size')
    ax.set_ylabel('Error metric')


def panel_error_metric(_test: Test):
    fig, ax = plt.subplots(1, 1, figsize=(2, 1.25), layout='compressed')

    _panel_error_metric(_test, ax)

    return fig, ax


class ComparisonData:

    def __init__(self,
                 sample_size: int,
                 beta_facts: List[float],
                 var_names: List[str],
                 self_sim_stats: Tuple[float, float],
                 comparison_errors: List[float]):
        self.sample_size = sample_size
        self.beta_facts = beta_facts
        self.var_names = var_names
        self.comparison_errors = comparison_errors
        self.self_sim_stats = self_sim_stats

    @staticmethod
    def load(fp: str):
        with open(fp, 'r') as f:
            data = json.load(f)

        beta_facts = [float(f) for f in data['beta_facts']]
        var_names = list(data['results'].keys())
        comparison_errors_vars = {n: [] for n in var_names}
        [[comparison_errors_vars[n].append(float(d[n])) for n in var_names] for d in data['comparison_error']]
        comparison_errors = [max([v[i] for v in comparison_errors_vars.values()]) for i in range(len(beta_facts))]
        return ComparisonData(sample_size=int(data['sample_size']),
                              beta_facts=beta_facts,
                              var_names=var_names,
                              self_sim_stats=(np.average(data['self_sim_evals']), np.std(data['self_sim_evals'])),
                              comparison_errors=comparison_errors)


def _panel_error_comparison(data: Dict[int, ComparisonData], axs):
    sizes = list(data.keys())
    for i, sz in enumerate(sizes):
        data_sz = data[sz]
        axs[i].scatter(data_sz.beta_facts, data_sz.comparison_errors)
        err = data_sz.self_sim_stats[1] * 3
        axs[i].fill_between(data_sz.beta_facts, data_sz.self_sim_stats[0] - err, data_sz.self_sim_stats[0] + err,
                            color='lightgray', alpha=0.5)
    for i, sz in enumerate(sizes):
        axs[i].set_yscale('log')
        axs[i].set_xticks([0.5, 1.0, 1.5, 2.0])
        axs[i].set_yticks([1E-1, 1E0])
        axs[i].set_title(f'Sample size: {sz}').set_fontstyle('italic')
    axs[0].set_ylabel('Error metric')


def panel_error_comparison(res_dir: str):
    fig, axs = plt.subplots(1, 4, figsize=(4.5, 0.75), sharey=True, layout='compressed')

    _panel_error_comparison(load_comparison_data(res_dir), axs)

    return fig, axs


def generate_panes(res_dir: str, output_dir: str = None):
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    fp_coinfection = os.path.join(res_dir, 'coinfection.json')
    fp_lorentz = os.path.join(res_dir, 'lorentz.json')
    fp_sir = os.path.join(res_dir, 'sir.json')

    fig_coinfection, axs_coinfection = panel_coinfection(load_test(fp_coinfection))
    fig_coinfection.savefig(os.path.join(output_dir, 'coinfection.svg'))
    fig_coinfection.savefig(os.path.join(output_dir, 'coinfection.png'))

    fig_lorentz, axs_lorentz = panel_lorentz(load_test(fp_lorentz))
    fig_lorentz.savefig(os.path.join(output_dir, 'lorentz.svg'))
    fig_lorentz.savefig(os.path.join(output_dir, 'lorentz.png'))

    fig_sir, axs_sir = panel_sir(load_test(fp_sir))
    fig_sir.savefig(os.path.join(output_dir, 'sir.svg'))
    fig_sir.savefig(os.path.join(output_dir, 'sir.png'))

    fig_error_metric, axs_error_metric = panel_error_metric(load_test(fp_sir))
    fig_error_metric.savefig(os.path.join(output_dir, 'error_metric.svg'))
    fig_error_metric.savefig(os.path.join(output_dir, 'error_metric.png'))

    fig, axs = panel_error_comparison(os.path.join(res_dir, 'test_comparison'))
    print(os.path.join(output_dir, 'figure_overview_comparison_data.svg'))
    fig.savefig(os.path.join(output_dir, 'figure_overview_comparison_data.svg'))
    fig.savefig(os.path.join(output_dir, 'figure_overview_comparison_data.png'))


_worker_data = [
    ('coinfection', 'coinfection', panel_coinfection),
    ('lorentz', 'lorentz', panel_lorentz),
    ('sir', 'sir', panel_sir),
    ('sir', 'error_metric', panel_error_metric)
]


def _worker_func(wid: int, res_dir: str, output_dir: str):
    _res_name, _output_name, _worker_func = _worker_data[wid]
    fp = os.path.join(res_dir, _res_name + '.json')
    fig, axs = _worker_func(load_test(fp))
    fig.savefig(os.path.join(output_dir, _output_name + '.svg'))
    fig.savefig(os.path.join(output_dir, _output_name + '.png'))


def generate_panes_par(res_dir: str, output_dir: str = None):
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    with mp.Pool() as p:
        job = p.starmap_async(_worker_func, [(i, res_dir, output_dir) for i in range(len(_worker_data))])

        fig, axs = panel_error_comparison(os.path.join(res_dir, 'test_comparison'))
        fig.savefig(os.path.join(output_dir, 'figure_overview_comparison_data.svg'))
        fig.savefig(os.path.join(output_dir, 'figure_overview_comparison_data.png'))

        job.wait()


def _load_test(fname: str, fp: str):
    return fname, Test.load(fp)


def load_comparison_data(res_dir: str):
    res_sizes_names = {int(f.replace('test_comparison_', '').replace('.json', '')): f
                       for f in os.listdir(res_dir) if 'test_comparison_' in f and f.endswith('.json')}
    res_sizes = list(res_sizes_names.keys())
    res_sizes.sort()
    return {sz: ComparisonData.load(os.path.join(res_dir, res_sizes_names[sz])) for sz in res_sizes}


def _load_comparison_data(res_dir: str, conn):
    conn.send(load_comparison_data(res_dir))


def generate_figure(res_dir: str, output_dir: str = None, preview=False):
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    print('Loading data:', res_dir)

    p_conn_1, p_conn_2 = mp.Pipe()
    p_data = mp.Process(target=_load_comparison_data, args=[os.path.join(res_dir, 'test_comparison'), p_conn_1])
    p_data.start()

    results_names = ['coinfection', 'lorentz', 'sir']
    tests = {}

    with mp.Pool(len(results_names)) as p:
        for fname, test in p.starmap(_load_test, [(rn, os.path.join(res_dir, rn + '.json')) for rn in results_names]):
            tests[fname] = test

    print('Generating plot...')

    fig = plt.figure(figsize=(4.72, 4.72), layout='constrained')
    gs = fig.add_gridspec(5, 5, height_ratios=(1, 1, 1, 1, 1))

    label_kwargs = dict(
        fontsize=10,
        fontproperties={'weight': 'bold'}
    )
    subplot_kwargs = dict(
        wspace=0.001,
        hspace=0.001,
    )

    print('... panel: coinfection')
    subfig_coinfection = fig.add_subfigure(gs[0, :])
    axs_coinfection = subfig_coinfection.subplots(1, 5, gridspec_kw=subplot_kwargs)
    _panel_coinfection(tests['coinfection'], axs_coinfection)
    subfig_coinfection.text(0.01, 0.99, 'A', ha='left', va='top', **label_kwargs)
    subfig_coinfection.supxlabel('Time', fontsize=plt.rcParams['axes.labelsize'])

    print('... panel: lorentz')
    subfig_lorentz = fig.add_subfigure(gs[1, :])
    axs_lorentz = subfig_lorentz.subplots(1, 3, gridspec_kw=subplot_kwargs)
    _panel_lorentz(tests['lorentz'], axs_lorentz)
    subfig_lorentz.text(0.01, 0.99, 'B', ha='left', va='top', **label_kwargs)
    subfig_lorentz.supxlabel('Time', fontsize=plt.rcParams['axes.labelsize'])

    print('... panel: sir')
    subfig_sir = fig.add_subfigure(gs[2:4, 0:3])
    axs_sir = subfig_sir.subplots(2, 2, sharex=True, gridspec_kw=subplot_kwargs)
    _panel_sir(tests['sir'], axs_sir)
    subfig_sir.text(0.01, 0.99, 'C', ha='left', va='top', **label_kwargs)
    subfig_sir.supxlabel('Time', fontsize=plt.rcParams['axes.labelsize'])

    print('... panel: error metric')
    subfig_error_metric = fig.add_subfigure(gs[2:4, 3:])
    ax_error_metric = subfig_error_metric.subplots(1, 1)
    _panel_error_metric(tests['sir'], ax_error_metric)
    subfig_error_metric.text(0.01, 0.99, 'D', ha='left', va='top', **label_kwargs)

    print('... panel: error comparison')
    subfig_error_comparison = fig.add_subfigure(gs[-1, :])
    ax_error_comparison = subfig_error_comparison.subplots(1, 3, sharey=True, gridspec_kw=subplot_kwargs)
    figure_overview_comparison_data = p_conn_2.recv()
    p_data.close()
    _panel_error_comparison(figure_overview_comparison_data, ax_error_comparison)
    subfig_error_comparison.text(0.03, 0.99, 'E', ha='left', va='top', **label_kwargs)
    subfig_error_comparison.supxlabel('Parameter ratio', fontsize=plt.rcParams['axes.labelsize'])

    if preview:
        fig.show()
        plt.show(block=True)
    else:
        fig.savefig(os.path.join(output_dir, 'figure_overview.png'))
        fig.savefig(os.path.join(output_dir, 'figure_overview.svg'))


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

        self.add_argument('-p', '--parallel',
                          required=False,
                          action='store_true',
                          help='Flag to process in parallel',
                          dest='par')

        self.add_argument('-g', '--generate-fig',
                          required=False,
                          action='store_true',
                          help='Generate full figure',
                          dest='gen')

        self.add_argument('-gp', '--generate-preview',
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
    def par(self):
        return self.parsed_args.par

    @property
    def gen(self):
        return self.parsed_args.gen

    @property
    def preview(self):
        return self.parsed_args.preview

    @property
    def debug(self):
        return self.parsed_args.debug


if __name__ == '__main__':
    pa = ArgParse()
    if pa.debug:
        DEBUG = True

    if pa.gen:
        generate_figure(res_dir=pa.res_dir, output_dir=pa.output_dir, preview=pa.preview)
    else:
        if pa.par:
            generate_panes_par(res_dir=pa.res_dir, output_dir=pa.output_dir)
        else:
            generate_panes(res_dir=pa.res_dir, output_dir=pa.output_dir)
