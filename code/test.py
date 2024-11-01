import os
from typing import List

import stochastic_models as sm
from stochastic_tests import Test

quiet = False


def assemble_test(model: sm.SBMLModel, 
                  t_fin: float, 
                  num_steps: int, 
                  trials: List[int], 
                  stochastic: bool):
    return Test(model=model, 
                t_fin=t_fin, 
                num_steps=num_steps, 
                sample_times=[t_fin / num_steps * i for i in range(0, num_steps + 1)], 
                trials=trials, 
                stochastic=stochastic)


def run_test(test: Test, fp: str = None):
    if os.path.isfile(fp):
        print('Output already exists. Doing nothing:', fp)
        return
    
    if not quiet:
        if fp is not None:
            print('Running test for output:', fp)

    test.run()
    if fp is not None:
        test.save(fp)
