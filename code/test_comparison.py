import argparse
import json
import numpy as np
import os
from typing import Dict, List, Optional, Tuple

import sim_lib

t_fin = 10.0
num_steps = 100
sample_times = [t_fin / num_steps * i for i in range(0, num_steps + 1)]
stochastic = False
sig_figs = 9
sampling_err_thresh = 0.001

param_name = 'beta'
param_dist_name = 'norm'
param_dist_args = 2E-6, 2E-7

beta_facts = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1, 1.5, 2.0]
sample_sizes = [100, 1000, 10000]

model_string_antimony = """
species S, I, R, V;

S -> I ; beta * S * V;
I -> R ; delta * I;
-> V  ; p * I - k * V;

S = 1E6;
I = 0.0;
R = 0.0;
V = 2.0;

beta = 2.0E-6;
k = 4.0;
delta = 1E0;
p = 25.0;
"""
var_names = ['S', 'I', 'R', 'V']


def extract_rr_time(_res):
    return _res[:, _res.colnames.index('time')]


def extract_rr_results(_res, _var_name: str):
    return _res[:, _res.colnames.index(f'[{_var_name}]')]


class ComparisonResult:

    def __init__(self, sample_size: int, exec_start=True):

        model_string = sim_lib.antimony_to_sbml(model_string_antimony)

        self.sample_size: int = sample_size
        self.beta_facts: List[float] = beta_facts
        self.t_fin: float = t_fin
        self.num_steps: int = num_steps
        self.stochastic: bool = stochastic
        self.sig_figs: int = sig_figs
        self.sampling_err_thresh: float = sampling_err_thresh
        self.results_times: Optional[np.ndarray] = None

        self.dist_info: List[Dict[str, Tuple[str, Tuple[float, float]]]] = []
        self.results: Dict[str, np.ndarray] = {}

        self.self_sim_evals: Optional[List[float]] = None
        self.self_sim: Optional[Tuple[float, float]] = None
        self.comparison_results: List[Dict[str, np.ndarray]] = []
        self.comparison_error: List[Dict[str, float]] = []
        self.comparison_pvals: List[Dict[str, float]] = []
        self.eval_info: Optional[Dict[str, List[Tuple[int, float]]]] = None

        if exec_start:
            self.results_times = extract_rr_time(sim_lib.exec_rr(
                sim_lib.make_rr(model_string, stochastic), t_fin, num_steps, stochastic
            ))
            self.results = {n: np.zeros((sample_size, num_steps)) for n in var_names} if exec_start else {}

            dist_info = {param_name: (param_dist_name, param_dist_args)}
            rr = sim_lib.make_rr(model_string, stochastic)
            for j, res in enumerate(sim_lib.exec_rr_batch(sample_size, rr, t_fin, num_steps, stochastic, dists=dist_info)):
                for name in var_names:
                    self.results[name][j, :] = sim_lib.round_arr_to_sigfigs(extract_rr_results(res, name), sig_figs)
            self.self_sim_evals = sim_lib.test_sampling(self.results, err_thresh=sampling_err_thresh)[0]
            self.self_sim = np.average(self.self_sim_evals), np.std(self.self_sim_evals, ddof=1)

            pval_sample_size = len(self.self_sim_evals)
            results_hsize = int(sample_size/2)
            results_half = {k: v[:results_hsize, :] for k, v in self.results.items()}
            results_ecfs, _, self.eval_info = sim_lib.find_ecfs(results_half)
            for f in beta_facts:
                dist_info = {param_name: (param_dist_name, (param_dist_args[0] * f, param_dist_args[1] * f))}
                rr = sim_lib.make_rr(model_string, stochastic)
                results_comp = {n: np.zeros((results_hsize, num_steps)) for n in var_names}
                for j, res in enumerate(sim_lib.exec_rr_batch(results_hsize, rr, t_fin, num_steps, stochastic, dists=dist_info)):
                    for name in var_names:
                        results_comp[name][j, :] = sim_lib.round_arr_to_sigfigs(extract_rr_results(res, name), sig_figs)

                self.dist_info.append(dist_info)
                self.comparison_results.append(results_comp)

                self.comparison_error.append(dict())
                self.comparison_pvals.append(dict())
                for n in var_names:
                    err_max = 0.0
                    for i in range(num_steps):
                        ecf_comp = sim_lib.ecf(results_comp[n][:, i],
                                               sim_lib.get_eval_info_times(*self.eval_info[n][i]))
                        err_max = max(err_max, sim_lib.ecf_compare(results_ecfs[n][i], ecf_comp))
                    self.comparison_error[-1][n] = err_max
                    if err_max < self.self_sim[0]:
                        pval = 1.0
                    else:
                        q2 = (pval_sample_size + 1) / pval_sample_size * self.self_sim[1] * self.self_sim[1]
                        lam2 = ((err_max - self.self_sim[0]) ** 2) / q2
                        pval = np.floor((pval_sample_size + 1) / pval_sample_size * ((pval_sample_size - 1) / lam2 + 1)) / (pval_sample_size + 1)
                        if pval > 1:
                            pval = 1.0
                    self.comparison_pvals[-1][n] = pval

    def to_json(self):
        if self.results_times is None:
            raise RuntimeError('Cannot export uninitialized data')
        return dict(
            sample_size=self.sample_size,
            beta_facts=self.beta_facts,
            t_fin=self.t_fin,
            num_steps=self.num_steps,
            stochastic=self.stochastic,
            sig_figs=self.sig_figs,
            sampling_err_thresh=self.sampling_err_thresh,
            results_times=self.results_times.tolist(),
            dist_info=self.dist_info,
            results={k: v.tolist() for k, v in self.results.items()},
            self_sim_evals=self.self_sim_evals,
            self_sim=self.self_sim,
            comparison_results=[{k: v.tolist() for k, v in e.items()} for e in self.comparison_results],
            comparison_error=self.comparison_error,
            comparison_pvals=self.comparison_pvals,
            eval_info=self.eval_info
        )

    @classmethod
    def from_json(cls, data):
        result = cls(int(data['sample_size']), exec_start=False)
        result.beta_facts = [float(f) for f in data['beta_facts']]
        result.t_fin = float(data['t_fin'])
        result.num_steps = int(data['num_steps'])
        result.stochastic = bool(data['stochastic'])
        result.sig_figs = int(data['sig_figs'])
        result.sampling_err_thresh = float(data['sampling_err_thresh'])
        result.results_times = np.asarray(data['results_times'], dtype=float)
        result.dist_info = [{k: (v[0], (float(v[1][0]), float(v[1][1]))) for k, v in e.items()} for e in data['dist_info']]
        result.results = {k: np.asarray(v, dtype=float) for k, v in data['results'].items()}
        result.self_sim_evals = [float(v) for v in data['self_sim_evals']]
        result.self_sim = float(data['self_sim'][0]), float(data['self_sim'][1])
        result.comparison_results = [{k: np.asarray(v, dtype=float) for k, v in e.items()}
                                     for e in data['comparison_results']]
        result.comparison_error = [{k: float(v) for k, v in e.items()} for e in data['comparison_error']]
        result.comparison_pvals = [{k: float(v) for k, v in e.items()} for e in data['comparison_pvals']]
        result.eval_info = {k: [(int(vv[0]), float(vv[1])) for vv in v] for k, v in data['eval_info'].items()}

        return result


def main(output_dir: str):
    for sz in sample_sizes:
        result = ComparisonResult(sz)
        with open(os.path.join(output_dir, f'test_comparison_{sz}.json'), 'w') as f:
            json.dump(result.to_json(), f, indent=4)


class ArgParser(argparse.ArgumentParser):

    def __init__(self):

        super().__init__()

        self.add_argument('-o', '--output-dir',
                          type=str,
                          dest='output_dir',
                          default=os.path.join(os.path.dirname(__file__), 'test_comparison'),
                          help='Data output directory')

        self.parsed_args = self.parse_args()

    @property
    def output_dir(self) -> str:
        return self.parsed_args.output_dir


if __name__ == '__main__':
    pa = ArgParser()
    if not os.path.isdir(pa.output_dir):
        os.mkdir(pa.output_dir)
    main(pa.output_dir)
