import argparse
import json
import os

try:
    from .stochastic_models import model_sir
    from . import stochastic_repro as sr
    from .stochastic_tests import Test
except ImportError:
    from stochastic_models import model_sir
    import stochastic_repro as sr
    from stochastic_tests import Test

param_name = 'beta'
param_dist_name = 'norm'
param_dist_args = 2E-6, 2E-7

beta_fact = [0.5, 0.75, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.5, 2.0]


def run(results_dir: str, output_dir: str):
    summary_data = dict(beta_fact=[],
                        results=[],
                        results_data=[])
    
    fp = os.path.join(results_dir, 'sir.json')
    test_baseline = Test.load(fp)

    trial_oi = test_baseline.trials[-1]

    test_kwargs = dict(t_fin=test_baseline.t_fin,
                       num_steps=test_baseline.num_steps,
                       trials=[trial_oi],
                       stochastic=test_baseline.stochastic,
                       sample_times=test_baseline.sample_times)
    
    for i in range(len(beta_fact)):
        bfact = beta_fact[i]
        
        print(f'bfact = {bfact}')

        test_i = Test(model=model_sir({param_name: (param_dist_name, (param_dist_args[0] * bfact, param_dist_args[1] * bfact))}), **test_kwargs)
        test_i.execute_stochastic()
        ecf2 = sr.generate_ecfs(test_i.sims_s, test_baseline.sample_times, test_baseline.model.results_names, [trial_oi], test_baseline.ecf_eval_info)
        res_i = sr.measure_ecf_diff_sets({trial_oi: test_baseline.ecf[trial_oi]}, ecf2)[trial_oi]
        results_i = {k: max([r[k] for r in res_i]) for k in test_baseline.model.results_names}

        print('results_i =', results_i)
        
        summary_data['beta_fact'].append(bfact)
        summary_data['results'].append(results_i)
        
        fp_name = f'figure_overview_comparison_data_{i}.json'
        fp = os.path.join(output_dir, fp_name)

        print(f'fp = {fp}')

        summary_data['results_data'].append(fp_name)
        test_i.save(fp)

    with open(os.path.join(output_dir, 'figure_overview_comparison_data.json'), 'w') as f:
        json.dump(summary_data, f, indent=4)


class ArgParse(argparse.ArgumentParser):
    
    def __init__(self):
        
        super().__init__()
        
        self.add_argument('-r', '--res-dir',
                          required=True,
                          type=str,
                          dest='results_dir',
                          help='Absolute path to the directory containing results')
        
        self.add_argument('-o', '--output-dir',
                          required=False,
                          type=str,
                          dest='output_dir',
                          default=os.path.dirname(__file__),
                          help='Output directory')
        
        self.parsed_args = self.parse_args()

    @property
    def results_dir(self):
        return self.parsed_args.results_dir

    @property
    def output_dir(self):
        return self.parsed_args.output_dir


if __name__ == '__main__':
    pa = ArgParse()
    run(results_dir=pa.results_dir, output_dir=pa.output_dir)
