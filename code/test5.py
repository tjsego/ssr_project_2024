import os
import sys

sys.path.append('/Users/timothy.sego/Desktop/Current/stochastic_repro')

import stochastic_models as sm
from test import assemble_test, run_test


def test_231205_0():
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'test_231205_0')

    run_test(assemble_test(model=sm.biomodels_2004140002(),
                           t_fin=1800.0,
                           num_steps=1800,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=True), 
             os.path.join(results_dir, 'biomodels_2004140002.json'))

    run_test(assemble_test(model=sm.biomodels_1805160001(),
                           t_fin=1200.0,
                           num_steps=1200,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=True), 
             os.path.join(results_dir, 'biomodels_1805160001.json'))

    run_test(assemble_test(model=sm.biomodels_2001130001(),
                           t_fin=10.0,
                           num_steps=1000,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=True), 
             os.path.join(results_dir, 'biomodels_2001130001.json'))


if __name__ == '__main__':

    test_231205_0()
