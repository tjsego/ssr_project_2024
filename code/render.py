import os
from typing import List

from stochastic_tests import Test

quiet = False
preview = True
dpi = 300

labels = ['deterministic',
          'distributions',
          'ecf', 
          'ecf_comparison',
          'ecf_diff_fits', 
          'ecf_diffs', 
          'stochastic',
          'ks_sampling',
          'ks_sampling_fits']


def render(test: Test, 
           render_dir: str, 
           prefix: str = None, 
           render_hooks: dict = None, 
           stages: List[str] = None, 
           ext: str = '.png'):
    if not os.path.isdir(render_dir):
        os.mkdir(render_dir)

    basename = lambda s: s if prefix is None else prefix + '_' + s
    if ext is None:
        ext = '.png'

    if not quiet:
        print('Render directory: ', render_dir)
        print('Extension       :', ext)
        if prefix is not None:
            print('Prefix          :', prefix)
        if render_hooks is not None:
            print('Render hooks    :', list(render_hooks.keys()))
        if stages is not None:
            print('Stages          :', stages)
    
    # Gather info for plotting ECFs with
    #   greatest difference in largest sample
    #   smallest ECF evaluation window
    #   largest ECF evaluation window
    times_oi = []

    idx = 0
    ks_max = None
    for i, el in enumerate(test.ecf_ks_stat[test.trials[-1]]):
        for ks in el.values():
            if ks_max is None or ks > ks_max:
                ks_max = ks
                idx = i
    times_oi.append((test.sample_times[idx], f'_greatest_ks_{idx}'))

    idx_min = 0
    idx_max = 0
    win_min = None
    win_max = None
    for i, el in enumerate(test.ecf_eval_info[test.trials[-1]]):
        for _, tfin in el.values():
            if win_min is None or tfin < win_min:
                win_min = tfin
                idx_min = i
            if win_max is None or tfin > win_max:
                win_max = tfin
                idx_max = i
    times_oi.append((test.sample_times[idx_min], f'_smallest_win_{idx_min}'))
    times_oi.append((test.sample_times[idx_max], f'_largest_win_{idx_max}'))

    stage = 'deterministic'
    if stages is None or stage in stages:
        if not quiet:
            print('   Stage        :', stage)
        fig, ax = test.plot_results_deterministic()
        if render_hooks is not None and stage in render_hooks.keys():
            render_hooks[stage](fig, ax)
        if preview:
            fig.show()
        else:
            fig.savefig(os.path.join(render_dir, basename(stage) + ext), dpi=dpi)

    stage = 'stochastic'
    if stages is None or stage in stages:
        if not quiet:
            print('   Stage        :', stage)
        fig, ax = test.plot_results_stochastic()
        if render_hooks is not None and stage in render_hooks.keys():
            render_hooks[stage](fig, ax)
        if preview:
            fig.show()
        else:
            fig.savefig(os.path.join(render_dir, basename(stage) + ext), dpi=dpi)

    stage = 'distributions'
    if stages is None or stage in stages:
        if not quiet:
            print('   Stage        :', stage)
        fig, ax = test.plot_distributions()
        if render_hooks is not None and stage in render_hooks.keys():
            render_hooks[stage](fig, ax)
        if preview:
            fig.show()
        else:
            fig.savefig(os.path.join(render_dir, basename(stage) + ext), dpi=dpi)

    stage = 'ecf'
    if stages is None or stage in stages:
        if not quiet:
            print('   Stage        :', stage)

        for t, name in times_oi:
            fig, ax = test.plot_ecf(t)
            if render_hooks is not None and stage in render_hooks.keys():
                render_hooks[stage](fig, ax)
            if preview:
                fig.show()
            else:
                fig.savefig(os.path.join(render_dir, basename(stage) + name + ext), dpi=dpi)
    
    stage = 'ecf_diffs'
    if stages is None or stage in stages:
        if not quiet:
            print('   Stage        :', stage)
        fig, ax = test.plot_ecf_diffs()
        if render_hooks is not None and stage in render_hooks.keys():
            render_hooks[stage](fig, ax)
        if preview:
            fig.show()
        else:
            fig.savefig(os.path.join(render_dir, basename(stage) + ext), dpi=dpi)
    
    stage = 'ecf_comparison'
    if stages is None or stage in stages:
        if not quiet:
            print('   Stage        :', stage)
        
        for t, name in times_oi:
            fig_r, ax_r, fig_i, ax_i = test.plot_ecf_comparison(t)
            if render_hooks is not None and stage in render_hooks.keys():
                render_hooks[stage](fig_r, ax_r)
                render_hooks[stage](fig_i, ax_i)
            if preview:
                fig_r.show()
                fig_i.show()
            else:
                fig_r.savefig(os.path.join(render_dir, basename(stage) + name + '_real' + ext), dpi=dpi)
                fig_i.savefig(os.path.join(render_dir, basename(stage) + name + '_imag' + ext), dpi=dpi)

    stage = 'ecf_diff_fits'
    if stages is None or stage in stages:
        if not quiet:
            print('   Stage        :', stage)
        test.generate_ecf_diff_fits()
        fig, ax = test.plot_ecf_diffs()
        if render_hooks is not None and 'ecf_diffs' in render_hooks.keys():
            render_hooks['ecf_diffs'](fig, ax)
        fig, ax = test.plot_ecf_diff_fits((fig, ax))
        if render_hooks is not None and stage in render_hooks.keys():
            render_hooks[stage](fig, ax)
        if preview:
            fig.show()
        else:
            fig.savefig(os.path.join(render_dir, basename(stage) + ext), dpi=dpi)

    stage = 'ks_sampling'
    if stages is None or stage in stages:
        if not quiet:
            print('   Stage        :', stage)
        fig, ax = test.plot_ks_sampling()
        if render_hooks is not None and stage in render_hooks.keys():
            render_hooks[stage](fig, ax)
        if preview:
            fig.show()
        else:
            fig.savefig(os.path.join(render_dir, basename(stage) + ext), dpi=dpi)

    stage = 'ks_sampling_fits'
    if stages is None or stage in stages:
        if not quiet:
            print('   Stage        :', stage)
        test.generate_ecf_sampling_fits()
        fig, ax = test.plot_ecf_sampling_fits(test.plot_ecf_sampling())
        if render_hooks is not None and stage in render_hooks.keys():
            render_hooks[stage](fig, ax)
        if preview:
            fig.show()
        else:
            fig.savefig(os.path.join(render_dir, basename(stage) + ext), dpi=dpi)
