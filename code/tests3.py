import os
import sys

sys.path.append('/Users/timothy.sego/Desktop/Current/stochastic_repro')

import stochastic_models as sm
from test import assemble_test, run_test


def test_231129_0():
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'test_231129_0')

    run_test(assemble_test(model=sm.model_tellurium_ex({'k1': ('norm', (0.1, 0.01))}),
                           t_fin=50.0,
                           num_steps=1000,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=False), 
             os.path.join(results_dir, 'tellurium_ex.json'))

    run_test(assemble_test(model=sm.model_tellurium_ex(mods={'k1': 0.1}),
                           t_fin=50.0,
                           num_steps=1000,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=True), 
             os.path.join(results_dir, 'tellurium_ex1.json'))


if __name__ == '__main__':

    test_231129_0()
