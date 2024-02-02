import multiprocessing as mp
import os

from stochastic_tests import Test
import render

_render_231117_0_column_width = 3.0
_render_231117_0_row_height = 2.0
_render_231117_0_ecf_sampling_fit_width = 6.0
_render_231117_0_ecf_sampling_fit_height = 4.0


def _render_231117_0_generate_render_hooks(num_names: int, num_trials: int):
    def _render_hook_generated_deterministic(fig, ax):
        fig.set_size_inches(_render_231117_0_column_width * num_names, _render_231117_0_row_height)

    def _render_hook_generated_distributions(fig, ax):
        fig.set_size_inches(_render_231117_0_column_width * num_names, _render_231117_0_row_height * num_trials)

    def _render_hook_generated_ecf(fig, ax):
        from matplotlib.legend import Legend

        fig.set_size_inches(_render_231117_0_column_width * num_names, _render_231117_0_row_height * 2)
        leg = [c for c in fig.get_children() if isinstance(c, Legend)]
        if len(leg) == 1:
            leg = leg[0]
            leg.set_bbox_to_anchor((1.3, 0.75))

    def _render_hook_generated_ecf_diff_fits(fig, ax):
        leg = fig.legend(labels=['Measurements'] + [f'Samples: {i}' for i in range(2, 8)])
        leg.set_bbox_to_anchor((1.3, 0.9))
        
        for i in range(num_names):
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')

    def _render_hook_generated_ecf_diffs(fig, ax):
        fig.set_size_inches(_render_231117_0_column_width * num_names, _render_231117_0_row_height)
        
        for i in range(num_names):
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')

    def _render_hook_generated_ecf_comparison(fig, ax):
        fig.set_size_inches(_render_231117_0_column_width * num_names, _render_231117_0_row_height * num_trials)

    def _render_hook_generated_stochastic(fig, ax):
        fig.set_size_inches(_render_231117_0_column_width * num_names, _render_231117_0_row_height * num_trials)

    def _render_hook_generated_ks_sampling(fig, ax):
        fig.set_size_inches(_render_231117_0_row_height * 4, _render_231117_0_row_height * num_trials)

    def _render_hook_generated_ks_sampling_fits(fig, ax):
        fig.set_size_inches(_render_231117_0_ecf_sampling_fit_width, _render_231117_0_ecf_sampling_fit_height)
        for c in ax.get_children():
            try:
                if c.get_marker() == 'o':
                    c.set_markersize(4)
            except AttributeError:
                pass
        
        ax.set_xscale('log')
        ax.set_yscale('log')

    return {
        'deterministic': _render_hook_generated_deterministic,
        'distributions': _render_hook_generated_distributions,
        'ecf': _render_hook_generated_ecf,
        'ecf_diff_fits': _render_hook_generated_ecf_diff_fits,
        'ecf_diffs': _render_hook_generated_ecf_diffs,
        'ecf_comparison': _render_hook_generated_ecf_comparison,
        'stochastic': _render_hook_generated_stochastic,
        'ks_sampling': _render_hook_generated_ks_sampling,
        'ks_sampling_fits': _render_hook_generated_ks_sampling_fits
    }

_render_231117_0_render_hooks = {
    'bistable': _render_231117_0_generate_render_hooks(2, 7),
    'coinfection': _render_231117_0_generate_render_hooks(5, 7),
    'lorentz': _render_231117_0_generate_render_hooks(3, 7),
    'nlpendulum_angle': _render_231117_0_generate_render_hooks(2, 7),
    'nlpendulum_param': _render_231117_0_generate_render_hooks(2, 7),
    'nlpendulum_speed': _render_231117_0_generate_render_hooks(2, 7),
    'oscillator': _render_231117_0_generate_render_hooks(2, 7),
    'pulse': _render_231117_0_generate_render_hooks(2, 7),
    'seir': _render_231117_0_generate_render_hooks(5, 7),
    'sir': _render_231117_0_generate_render_hooks(4, 7)
}


def _render_231117_0(basename: str):
    render.preview = False
    render.quiet = False
    render.dpi = 600

    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'test_231117_0')
    test = Test.load(os.path.join(results_dir, basename + '.json'))
    render_hooks = _render_231117_0_render_hooks[basename] if basename in _render_231117_0_render_hooks.keys() else None
    render.render(test=test, render_dir=os.path.join(results_dir, 'renders'), prefix=basename, render_hooks=render_hooks)


def render_231117_0():
    basenames = ['bistable', 
                 'coinfection',
                 'lorentz',
                 'nlpendulum_angle',
                 'nlpendulum_param',
                 'nlpendulum_speed',
                 'oscillator',
                 'pulse',
                 'seir',
                 'sir']
    workers = []
    for bn in basenames:
        workers.append(mp.Process(target=_render_231117_0, args=(bn,)))
    [w.start() for w in workers]
    [w.join() for w in workers]


if __name__ == '__main__':
    render_231117_0()
