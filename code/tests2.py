import os
import sys

sys.path.append('/Users/timothy.sego/Desktop/Current/stochastic_repro')

import stochastic_models as sm
from test import assemble_test, run_test


def test_231121_0():
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'test_231121_0')

    run_test(assemble_test(model=sm.model_bistable2({'a': ('norm', (1.0, 0.25))}),
                           t_fin=10.0,
                           num_steps=1000,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=False), 
             os.path.join(results_dir, 'bistable2.json'))


if __name__ == '__main__':

    test_231121_0()
