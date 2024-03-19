import json
from time import time_ns

from stochastic_models import AntimonyModel
from stochastic_tests import Test
import sim_lib


def model_string(num_variables: int, rate_cf: float):
    if num_variables < 2:
        raise ValueError('Must use two or more variables')

    model_string_expressions = [f'->V{i} ; sin(a * time);' for i in range(num_variables)]
    model_string = '\n'.join(model_string_expressions)

    model_string_params = [f'a = {rate_cf};']
    model_string += '\n'.join([''] + model_string_params)

    return model_string, [f'V{i}' for i in range(num_variables)], ['a']


num_reps = 10
rate = 0.1
stdev_cf = 0.25
model_sizes = [2, 3, 4, 5, 10]

t_fin = 50.0
num_steps = 100
test_kwargs = dict(t_fin=t_fin,
                   num_steps=num_steps,
                   trials=[100, 1000, 10000],
                   sample_times=[t_fin / num_steps * i for i in range(0, num_steps + 1)],
                   stochastic=False)

if __name__ == '__main__':

    _ = sim_lib.start_pool()

    runtime_tests = {ms: [] for ms in model_sizes}
    for model_size in model_sizes:
        ms, var_names, param_names = model_string(model_size, rate)

        for i in range(num_reps):
            print(f'Running size {model_size} replicate {i+1} of {num_reps}')

            test = Test(model=AntimonyModel(ms,
                                            var_names,
                                            param_dists={param_names[0]: ('norm', (rate, rate * stdev_cf))}),
                        **test_kwargs)
            test.execute_deterministic()
            test.execute_stochastic()
            test.find_ecfs()
            test.measure_ecf_diffs()

            time_start = time_ns()
            test.test_sampling(err_thresh=1E-3)
            time_elapsed = time_ns() - time_start
            runtime_tests[model_size].append(time_elapsed)

            print(f'... Time elapsed: {time_elapsed / 1E9} s')

            test.generate_ecf_sampling_fits()

            # Save a representative sample
            if i == 0:
                fp = f'perf_compare_size_1_{i+1}.json'
                print('Saving output:', fp)
                test.save(fp)

    fp_summary = 'perf_compare_size_1_inputs.json'
    print('Outputting summary:', fp_summary)
    with open(fp_summary, 'w') as f:
        output_data = dict(
            rate=rate,
            stdev_cf=stdev_cf,
            runtimes=runtime_tests,
            data={model_sizes[i]: f'perf_compare_size_1_{i+1}.json' for i in range(len(model_sizes))}
        )
        json.dump(output_data, f, indent=4)
