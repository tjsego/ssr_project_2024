import os

import stochastic_models as sm
from test import assemble_test, run_test


param_nominal_vals = {'beta': 2.8E-6,
                      'k': 4.0,
                      'delta': 0.89,
                      'p': 25.1,
                      'c': 28.4,
                      'r': 27.0,
                      'KP': 2.3E8,
                      'gammaMA': 1.35E-4,
                      'psi': 1.2E-8,
                      't0': 7.0}
stdev_fact = 0.25
param_range_fact = 0.5


def test_231129_1():
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'test_231129_1')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    run_test(assemble_test(model=sm.model_coinfection({k: ('norm', (v, v * stdev_fact)) for k, v in param_nominal_vals.items()}),
                           t_fin=18.0,
                           num_steps=100,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=False), 
             os.path.join(results_dir, 'paramvar_10.json'))

    run_test(assemble_test(model=sm.model_coinfection({k: ('uniform', (v * (1-param_range_fact), v * (1+param_range_fact))) for k, v in param_nominal_vals.items()}),
                           t_fin=18.0,
                           num_steps=100,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=False), 
             os.path.join(results_dir, 'paramvar_10_2.json'))


if __name__ == '__main__':

    test_231129_1()
