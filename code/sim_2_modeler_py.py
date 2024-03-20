import numpy as np
import os
import json
import multiprocessing as mp
from typing import Any, Dict, List

import sim_lib

# if __name__ == '__main__':
#     mp.set_start_method('spawn', force=True)

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
err_thresh = 0.075
stochastic = False
t_fin = 10.0
num_steps = 101
param_dist = {'beta': ('norm', (2.0E-6, 0.2E-6))}
sig_figs = 6  # For comparing to COPASI default data output
sampling_err_thresh = 0.001


def extend_arrs(_arr: np.ndarray, _num_extend: int):
    if _num_extend <= 0:
        raise ValueError

    result = np.zeros((_arr.shape[0] + _num_extend, _arr.shape[1]), dtype=float)
    result[:_arr.shape[0], :] = _arr
    return result


def extract_rr_time(_res):
    return _res[:, _res.colnames.index('time')]


def extract_rr_results(_res, _var_name: str):
    return _res[:, _res.colnames.index(f'[{_var_name}]')]


class SimulationReport:

    def __init__(self,
                 model_string: str,
                 var_names: List[str],
                 stochastic: bool,
                 t_fin: float,
                 num_steps: int,
                 err_thresh: float,
                 sampling_err_thresh: float,
                 sig_figs: int,
                 results_times: np.ndarray,
                 results: Dict[str, np.ndarray],
                 dists: Dict[str, Any] = None):
        
        self.model_string = model_string
        self.var_names = var_names
        self.stochastic = stochastic
        self.t_fin = t_fin
        self.num_steps = num_steps
        self.err_thresh = err_thresh
        self.sampling_err_thresh = sampling_err_thresh
        self.sig_figs = sig_figs
        self.dists = dists

        self.results_times = results_times
        self.results = results

        self.stat_hist = []
        self.err_hist = []
        self.ks_stats_samp_hist = []

    def to_json(self):
        return dict(
            model_string=self.model_string, 
            var_names=self.var_names, 
            stochastic=self.stochastic, 
            t_fin=self.t_fin, 
            num_steps=self.num_steps, 
            err_thresh=self.err_thresh, 
            sampling_err_thresh=self.sampling_err_thresh, 
            sig_figs=self.sig_figs,
            dists=[d.to_json() for d in self.param_dists] if self.param_dists is not None else None,
            results_times=self.results_times.tolist(),
            results={n: v.tolist() for n, v in self.results.items()},
            stat_hist=self.stat_hist,
            err_hist=self.err_hist,
            ks_stats_samp_hist=self.ks_stats_samp_hist,
        )

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        if data['dists'] is None:
            dists = None
        else:
            dists = {}
            for pd in data['dists']:
                d = sim_lib.ParamDist.from_json(pd)
                dists[d.param_name] = (d.dist_name, d.dist_params)

        report = SimulationReport(
            model_string=data['model_string'],
            var_names=data['var_names'],
            stochastic=bool(data['stochastic']),
            t_fin=float(data['t_fin']),
            num_steps=int(data['num_steps']),
            err_thresh=float(data['err_thresh']),
            sampling_err_thresh=float(data['sampling_err_thresh']),
            sig_figs=int(data['sig_figs']),
            results_times=np.asarray(data['results_times'], dtype=float),
            results={n: np.asarray(v, dtype=float) for n, v in data['results'].items()},
            dists=dists
        )
        report.stat_hist = [(int(t[0]), float(t[1]), float(t[2])) for t in data['stat_hist']]
        report.err_hist = [(int(t[0]), float(t[1])) for t in data['err_hist']]
        report.ks_stats_samp_hist = data['ks_stats_samp_hist']

        return report

    def __reduce__(self):
        return SimulationReport.from_json, (self.to_json(),)

    @property
    def num_results(self) -> int:
        return self.results[self.var_names[-1]].shape[0]

    @property
    def param_dists(self) -> List[sim_lib.ParamDist]:
        return [sim_lib.ParamDist(param_name=n, dist_name=t[0], dist_params=t[1]) for n, t in self.dists.items()]

    @property
    def results_export(self):
        sample_size = int(self.num_results/2)
        return {n: self.results[n][:sample_size, :] for n in self.var_names}

    def generate_metadata(self) -> sim_lib.Metadata:
        sample_size = int(self.num_results/2)
        ecf_evals, _, ecf_eval_info = sim_lib.find_ecfs(self.results_export)
        return sim_lib.Metadata(sample_size=sample_size,
                                simulator='deterministic',
                                ks_stat_mean=np.mean(self.ks_stats_samp_hist[-1]),
                                ks_stat_stdev=np.std(self.ks_stats_samp_hist[-1]),
                                sample_times=self.results_times,
                                ecf_evals=ecf_evals,
                                ecf_eval_info=ecf_eval_info,
                                param_dists=[sim_lib.ParamDist(param_name=n, dist_name=t[0], dist_params=t[1]) for n, t in self.dists.items()],
                                sig_figs=self.sig_figs)

    @staticmethod
    def simulate(model_string: str,
                 var_names: List[str],
                 stochastic: bool,
                 t_fin: float,
                 num_steps: int,
                 err_thresh: float,
                 sampling_err_thresh: float,
                 sig_figs: int,
                 dists: Dict[str, Any] = None,
                 num_samples_incr=100):
        
        rr = sim_lib.make_rr(model_string, stochastic)
        report = SimulationReport(model_string=model_string,
                                  var_names=var_names,
                                  stochastic=stochastic,
                                  t_fin=t_fin,
                                  num_steps=num_steps,
                                  err_thresh=err_thresh,
                                  sampling_err_thresh=sampling_err_thresh,
                                  sig_figs=sig_figs,
                                  results_times=extract_rr_time(sim_lib.exec_rr(rr, t_fin, num_steps, stochastic)),
                                  results={name: np.zeros((num_samples_incr, num_steps), dtype=float) for name in var_names},
                                  dists=dists)
        rr.resetAll()

        idx_current = 0
        iter_current = 0
        err_current = min(1, err_thresh + 1)

        while err_current >= err_thresh:

            if iter_current > 0:
                print(f'Iteration {iter_current} ({idx_current}): {report.stat_hist[-1][1]}, {report.stat_hist[-1][2]} ({err_thresh})')
            else:
                print(f'Iteration {iter_current} ({idx_current}): ({err_thresh})')

            num_samples_incr_curr = num_samples_incr

            if iter_current > 2:
                # Try to jump ahead
                x_data = []
                y_data = []
                for el in report.stat_hist:
                    x_data.append(el[0])
                    y_data.append(el[1] + 3 * el[2])
                fit_num_samples = int(sim_lib.recommend(x_data, y_data, err_thresh))
                fit_num_samples += fit_num_samples % 2
                if fit_num_samples < idx_current:
                    num_samples_incr_curr = num_samples_incr
                else:
                    num_samples_incr_curr = fit_num_samples - idx_current
                    print(f'Recommended sample size: {num_samples_incr_curr}')
                    
                    # Limit, in case we get something crazy from a poor fit
                    num_samples_incr_curr = max(min(num_samples_incr_curr, 5000), num_samples_incr)
                    
                    print(f'Jumping ahead with fitted sample size increment: {num_samples_incr_curr}')

            if iter_current > 0:
                # Extend data storage
                for name in var_names:
                    report.results[name] = extend_arrs(report.results[name], num_samples_incr_curr)

            print(f'Working {num_samples_incr_curr} jobs')
            idx_received = 0
            for res in sim_lib.exec_rr_batch(num_samples_incr_curr, rr, t_fin, num_steps, stochastic, dists=dists):
                for name in var_names:
                    report.results[name][idx_current+idx_received, :] = sim_lib.round_arr_to_sigfigs(extract_rr_results(res, name), report.sig_figs)
                idx_received += 1
            if idx_received != num_samples_incr_curr:
                print(f'Received {idx_received} results, though {num_samples_incr_curr} were requested.')
            else:
                print('All results received.')
            idx_current += idx_received
            
            print('Testing sampling')

            report.ks_stats_samp_hist.append(sim_lib.test_sampling(report.results, err_thresh=sampling_err_thresh)[0])
            report.stat_hist.append((idx_current, np.average(report.ks_stats_samp_hist[-1]), np.std(report.ks_stats_samp_hist[-1])))
            err_current = report.stat_hist[-1][1] + 3 * report.stat_hist[-1][2]
            report.err_hist.append((iter_current, err_current))

            print(f'Iteration {iter_current} ({idx_current}): {report.stat_hist[-1][1]}, {report.stat_hist[-1][2]} ({err_thresh})')
            iter_current += 1
        
        return report


def export_data(report: SimulationReport, results_dir: str):

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    fp = os.path.join(results_dir, 'sim_modeler.json')
    with open(fp, 'w') as f:
        json.dump(report.generate_metadata().to_json(), f, indent=4)

    # Record data for later reuse

    with open(os.path.join(results_dir, 'simdata_modeler.json'), 'w') as f:
        json.dump(report.to_json(), f, indent=4)


def do_sig_figs_6():
    if sim_lib.get_pool() is None:
        sim_lib.start_pool()

    report = SimulationReport.simulate(sim_lib.antimony_to_sbml(model_string_antimony),
                                       var_names,
                                       stochastic,
                                       t_fin,
                                       num_steps,
                                       err_thresh,
                                       sampling_err_thresh,
                                       sig_figs,
                                       param_dist)
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'workflow_sim_2_py', 'modeler_6')
    export_data(report, results_dir)


def do_sig_figs_9():
    if sim_lib.get_pool() is None:
        sim_lib.start_pool()

    report = SimulationReport.simulate(sim_lib.antimony_to_sbml(model_string_antimony),
                                       var_names,
                                       stochastic,
                                       t_fin,
                                       num_steps,
                                       err_thresh,
                                       sampling_err_thresh,
                                       9,
                                       param_dist)
    
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'workflow_sim_2_py', 'modeler_9')
    export_data(report, results_dir)


def main():

    do_sig_figs_6()
    do_sig_figs_9()


if __name__ == '__main__':
    main()
