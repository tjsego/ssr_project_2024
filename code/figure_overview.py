import argparse
import json
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import os

try:
    from .stochastic_tests import Test
except ImportError:
    from stochastic_tests import Test

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8.0
# plt.rcParams['figure.edgecolor'] = 'black'
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
            ax.plot(_test.trials, [Test.ecf_diff_fit_func(n, *data_f) for n in _test.trials], label=f'Sample size {_test.trials[i+1]}')
    
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


def _panel_error_comparison(data, axs):
    model_names = ['S', 'I', 'R', 'V']
    results_data = {k: [float(d[k]) for d in data['results']] for k in model_names}

    for i in range(len(model_names)):
        axs[i].scatter(data['beta_fact'], results_data[model_names[i]])
        axs[i].set_yscale('log')
        axs[i].set_xticks([0.5, 1.0, 1.5, 2.0])
        axs[i].set_yticks([1E-1, 1E0])
        axs[i].set_title(model_names[i]).set_fontstyle('italic')


def panel_error_comparison(fp: str):
    with open(fp, 'r') as f:
        data = json.load(f)

    fig, axs = plt.subplots(1, 4, figsize=(4.5, 0.75), sharey=True, layout='compressed')
    
    _panel_error_comparison(data, axs)
    
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
    
    fig, axs = panel_error_comparison(os.path.join(res_dir, 'figure_overview_comparison_data.json'))
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

        fig, axs = panel_error_comparison(os.path.join(res_dir, 'figure_overview_comparison_data.json'))
        fig.savefig(os.path.join(output_dir, 'figure_overview_comparison_data.svg'))
        fig.savefig(os.path.join(output_dir, 'figure_overview_comparison_data.png'))
        
        job.wait()


def _load_test(fname: str, fp: str):
    return fname, Test.load(fp)


def _load_comparison_data(fp: str, conn):
    with open(fp, 'r') as f:
        data = json.load(f)
    conn.send(data)


def generate_figure(res_dir: str, output_dir: str = None, preview=False):
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    print('Loading data:', res_dir)

    p_conn_1, p_conn_2 = mp.Pipe()
    p_data = mp.Process(target=_load_comparison_data, args=[os.path.join(res_dir, 'figure_overview_comparison_data.json'), p_conn_1])
    p_data.start()

    results_names = ['coinfection', 'lorentz', 'sir']
    tests = {}

    with mp.Pool(len(results_names)) as p:
        for fname, test in p.starmap(_load_test, [(rn, os.path.join(res_dir, rn + '.json')) for rn in results_names]):
            tests[fname] = test

    figure_overview_comparison_data = p_conn_2.recv()
    p_data.close()
    
    print('Generating plot...')
    
    fig = plt.figure(figsize=(4.72, 4.72), layout='constrained')
    gs = fig.add_gridspec(5, 5)

    label_kwargs = dict(
        fontsize=10,
        fontproperties={'weight': 'bold'}
    )

    print('... panel: coinfection')
    subfig_coinfection = fig.add_subfigure(gs[0, :])
    axs_coinfection = subfig_coinfection.subplots(1, 5)
    _panel_coinfection(tests['coinfection'], axs_coinfection)
    subfig_coinfection.text(0.01, 0.85, 'A', **label_kwargs)

    print('... panel: lorentz')
    subfig_lorentz = fig.add_subfigure(gs[1, :])
    axs_lorentz = subfig_lorentz.subplots(1, 3)
    _panel_lorentz(tests['lorentz'], axs_lorentz)
    subfig_lorentz.text(0.01, 0.95, 'B', **label_kwargs)

    print('... panel: sir')
    subfig_sir = fig.add_subfigure(gs[2:4, 0:3])
    axs_sir = subfig_sir.subplots(2, 2, sharex=True)
    _panel_sir(tests['sir'], axs_sir)
    subfig_sir.text(0.01, 0.95, 'C', **label_kwargs)

    print('... panel: error metric')
    subfig_error_metric = fig.add_subfigure(gs[2:4, 3:])
    ax_error_metric = subfig_error_metric.subplots(1, 1)
    _panel_error_metric(tests['sir'], ax_error_metric)
    subfig_error_metric.text(0.01, 0.95, 'D', **label_kwargs)

    print('... panel: error comparison')
    subfig_error_comparison = fig.add_subfigure(gs[-1, :])
    ax_error_comparison = subfig_error_comparison.subplots(1, 4, sharey=True)
    subfig_error_comparison.suptitle('Parameter ratio', y=0.1, verticalalignment='top')
    _panel_error_comparison(figure_overview_comparison_data, ax_error_comparison)
    subfig_error_comparison.text(0.01, 0.95, 'E', **label_kwargs)
    
    fig.get_layout_engine().set(h_pad=0.1)

    if preview:
        fig.show()
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