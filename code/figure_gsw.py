import argparse
import json
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
from PIL import Image

try:
    from . import gsw_lib
    from . import sim_lib
except ImportError:
    import gsw_lib
    import sim_lib

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

# Assuming interpreter can find this
gsw_netlogo_screenshot = 'gsw_netlogo_screenshot.png'

prefices = ['Grass', 'Sheep', 'Wolves']
implementations = ['NetLogo', 'MATLAB', 'Python']
size_test = [100, 500, 1000, 5000, 10000, 15000, 20000]

# Hack algorithm into library
sim_lib.known_sim_algs.append('GSW')


def load_metadata(results_dir: str):
    impl_map = {'NetLogo': 'nl',
                'MATLAB': 'ml',
                'Python': 'py'}
    
    data = {n: {} for n in implementations}
    
    for name_impl in implementations:
        for sz in size_test:
            fp = os.path.join(results_dir, f'gsw_{impl_map[name_impl]}_{sz}.json')
            with open(fp, 'r') as f:
                data[name_impl][sz] = sim_lib.Metadata.from_json(json.load(f))
    return data


def plot_spatial_screenshot(ax):
    im = Image.open(gsw_netlogo_screenshot)
    ax.imshow(im)


def plot_trajectories(axs, results_time, data_netlogo, data_matlab, data_python):
    data_impl = {'NetLogo': data_netlogo,
                 'MATLAB': data_matlab,
                 'Python': data_python}
    scilimits = {'Grass': (1, 4),
                 'Sheep': (1, 3),
                 'Wolves': (1, 2)}
    xlabels = {'Sheep': [2E3, 4E3, 6E3]}

    for i, name_var in enumerate(prefices):
        for j, name_impl in enumerate(implementations):
            p_data = data_impl[name_impl][name_var]
            for repl_num in range(p_data.shape[0]):
                axs[i][j].plot(results_time, p_data[repl_num, :], alpha=0.1, color='gray')
    for i, name_var in enumerate(prefices):
        axs[i][0].set_ylabel(name_var).set_fontstyle('italic')
        axs[i][0].ticklabel_format(axis='y', style='sci', useOffset=False, scilimits=scilimits[name_var])
        if name_var in xlabels.keys():
            for ax in axs[i]:
                ax.set_yticks(xlabels[name_var], minor=False)
    for i, name_impl in enumerate(implementations):
        axs[0][i].set_title(name_impl, y=1.0, pad=10.0).set_fontstyle('italic')


def plot_selfsim(axs, metadata):
    for i, impl_name in enumerate(implementations):
        avg = [metadata[impl_name][sz].ks_stat_mean for sz in size_test]
        std = [metadata[impl_name][sz].ks_stat_stdev for sz in size_test]
        axs[i].errorbar(size_test, avg, yerr=std, marker='o')
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_title(impl_name).set_fontstyle('italic')

    axs[0].set_ylabel('EFECT error')


def plot_comparison(axs, metadata):
    plot_scenarios = [
        ('NetLogo', 'MATLAB'), 
        ('NetLogo', 'Python'),
        ('MATLAB', 'Python')
    ]
    
    for i, ps in enumerate(plot_scenarios):
        error_metric = []
        for sz in size_test:
            mdata = metadata[ps[0]][sz], metadata[ps[1]][sz]
            err = 0.0
            for t_idx in range(len(mdata[0].sample_times)):
                for name_var in prefices:
                    ecf0 = mdata[0].ecf_evals[name_var][t_idx]
                    ecf1 = mdata[1].ecf_evals[name_var][t_idx]
                    err = max(err, sim_lib.ecf_compare(ecf0, ecf1))
            error_metric.append(err)

        axs[i].scatter(size_test, error_metric)
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_yticks([1, 2], minor=False)
        axs[i].set_yticks([], minor=True)
        axs[i].set_title(f'{ps[0]}/{ps[1]}').set_fontstyle('italic')
    
    axs[0].set_ylabel('EFECT error')


def generate_figure(results_dir: str, output_dir: str = None, preview=False):
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    print('Loading data:', results_dir)
    sim_lib.start_pool()

    results_time, data_nl = gsw_lib.load_data(prefices, results_dir + '/netlogo', 1, max(size_test), pool=sim_lib.get_pool())
    _, data_ml = gsw_lib.load_data(prefices, results_dir + '/matlab', 1, max(size_test), pool=sim_lib.get_pool())
    _, data_py = gsw_lib.load_data(prefices, results_dir + '/python', 0, max(size_test), pool=sim_lib.get_pool())
    
    metadata = load_metadata(os.path.join(results_dir, 'analysis_data'))
    
    print('Generating plot...')

    fig = plt.figure(figsize=(4.72, 4.72), layout='constrained')
    gs = fig.add_gridspec(3, 2, height_ratios=(2, 1, 1))

    label_kwargs = dict(
        fontsize=10,
        fontproperties={'weight': 'bold'}
    )
    subplot_kwargs = dict(
        wspace=0.001,
        hspace=0.001,
    )

    gs_top = gridspec.GridSpecFromSubplotSpec(3, 6, subplot_spec=gs[0, :])

    subfig_spatial = fig.add_subfigure(gs_top[:, :3])
    ax_spatial = subfig_spatial.subplots(1, 1, subplot_kw={'xticks': [], 'yticks': []}, gridspec_kw=subplot_kwargs)
    plot_spatial_screenshot(ax_spatial)
    subfig_spatial.text(0.05, 0.95, 'A', ha='left', va='top',
                        bbox=dict(edgecolor='black', facecolor='white'), **label_kwargs)

    subfig_trajs = fig.add_subfigure(gs_top[:, 3:])
    axs_trajs = subfig_trajs.subplots(3, 3, sharex=True, sharey='row', gridspec_kw=subplot_kwargs)
    plot_trajectories(axs_trajs, results_time, data_nl, data_ml, data_py)
    subfig_trajs.text(0.01, 0.99, 'B', ha='left', va='top', **label_kwargs)
    subfig_trajs.supxlabel('Time', fontsize=plt.rcParams['axes.labelsize'])
    subfig_trajs.align_labels()

    subfig_self = fig.add_subfigure(gs[1, :])
    axs_self = subfig_self.subplots(1, 3, sharey=True, gridspec_kw=subplot_kwargs)
    plot_selfsim(axs_self, metadata)
    subfig_self.text(0.01, 0.99, 'C', ha='left', va='top', **label_kwargs)
    subfig_self.supxlabel('Sample size', fontsize=plt.rcParams['axes.labelsize'])

    subfig_comp = fig.add_subfigure(gs[2, :])
    axs_comp = subfig_comp.subplots(1, 3, sharey=True, gridspec_kw=subplot_kwargs)
    plot_comparison(axs_comp, metadata)
    subfig_comp.text(0.01, 0.99, 'D', ha='left', va='top', **label_kwargs)
    subfig_comp.supxlabel('Sample size', fontsize=plt.rcParams['axes.labelsize'])

    if preview:
        fig.show()
        plt.show(block=True)
    else:
        fig.savefig(os.path.join(output_dir, 'figure_gsw.png'))
        fig.savefig(os.path.join(output_dir, 'figure_gsw.svg'))


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


if __name__ == '__main__':
    pa = ArgParse()
    generate_figure(results_dir=pa.res_dir, output_dir=pa.output_dir, preview=pa.preview)
