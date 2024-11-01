import os

import stochastic_models as sm
from test import assemble_test, run_test


def test_231117_0():
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'test_231117_0')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    run_test(assemble_test(model=sm.model_sir({'beta': ('norm', (2.0E-6, 0.2E-6))}), 
                           t_fin=10.0, 
                           num_steps=1000, 
                           trials=[10, 50, 100, 500, 1000, 5000, 10000], 
                           stochastic=False), 
             os.path.join(results_dir, 'sir.json'))

    run_test(assemble_test(model=sm.model_bistable({'y': ('uniform', (0.5, 1.0))}),
                           t_fin=10.0,
                           num_steps=1000,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=False), 
             os.path.join(results_dir, 'bistable.json'))

    run_test(assemble_test(model=sm.model_lorentz(),
                           t_fin=5.0,
                           num_steps=1000,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=True), 
             os.path.join(results_dir, 'lorentz.json'))

    run_test(assemble_test(model=sm.model_oscillator({'t0': ('norm', (0, 1E0))}),
                           t_fin=10.0,
                           num_steps=1000,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=False),
             os.path.join(results_dir, 'oscillator.json'))

    test_nlpendulum_kwargs = dict(t_fin=10.0, 
                                  num_steps=1000, 
                                  trials=[10, 50, 100, 500, 1000, 5000, 10000], 
                                  stochastic=False)

    run_test(assemble_test(model=sm.model_nlpendulum({'v': ('norm', (0.0, 0.25))}), **test_nlpendulum_kwargs), 
             os.path.join(results_dir, 'nlpendulum_speed.json'))
    run_test(assemble_test(model=sm.model_nlpendulum({'t': ('norm', (0.0, 1.0))}), **test_nlpendulum_kwargs), 
             os.path.join(results_dir, 'nlpendulum_angle.json'))
    run_test(assemble_test(model=sm.model_nlpendulum({'a': ('norm', (1.0, 0.25))}), **test_nlpendulum_kwargs), 
             os.path.join(results_dir, 'nlpendulum_param.json'))

    run_test(assemble_test(model=sm.model_coinfection({'beta': ('norm', (2.8E-6, 0.28E-6 * 2))}),
                           t_fin=18.0,
                           num_steps=1000,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=False),
             os.path.join(results_dir, 'coinfection.json'))

    run_test(assemble_test(model=sm.model_pulse(),
                           t_fin=20.0,
                           num_steps=2000,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=False),
             os.path.join(results_dir, 'pulse.json'))

    run_test(assemble_test(model=sm.model_seir(),
                           t_fin=100.0,
                           num_steps=1000,
                           trials=[10, 50, 100, 500, 1000, 5000, 10000],
                           stochastic=True), 
             os.path.join(results_dir, 'seir.json'))


if __name__ == '__main__':

    test_231117_0()
