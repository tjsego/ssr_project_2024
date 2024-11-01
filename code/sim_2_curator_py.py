import numpy as np
import os
import json
from typing import Dict, List

import sim_lib


def get_modeler_metadata(res_dir: str):
    with open(os.path.join(res_dir, 'sim_modeler.json'), 'r') as f:
        return sim_lib.Metadata.from_json(json.load(f))


def get_curator_results(res_dir, var_names, num_results, num_steps, sig_figs):
    results = dict()
    sample_times = np.zeros((num_steps,))

    for name in var_names:
        sample_times, results_name = sim_lib.load_results_copasi(os.path.join(res_dir, f'sim_curator_2_{name}.txt'),
                                                                 num_results,
                                                                 num_steps)
        results[name] = sim_lib.round_arr_to_sigfigs(results_name, sig_figs)

    return sample_times, results


class CuratorAnalysis:

    def __init__(self,
                 var_names: List[str],
                 num_steps: int,
                 sampling_err_thresh: float,
                 sig_figs: int,
                 results_times: np.ndarray,
                 results: Dict[str, np.ndarray],
                 ks_stats_samp=None,
                 err_max=None):

        self.var_names = var_names
        self.num_steps = num_steps
        self.sampling_err_thresh = sampling_err_thresh
        self.sig_figs = sig_figs
        self.results_times = results_times
        self.results = results

        if ks_stats_samp is None:
            self.ks_stats_samp = sim_lib.test_sampling(results, err_thresh=sampling_err_thresh)[0]
        else:
            self.ks_stats_samp = ks_stats_samp
        self.err_max = err_max

    def copy(self):
        return CuratorAnalysis(self.var_names,
                               self.num_steps,
                               self.sampling_err_thresh,
                               self.sig_figs,
                               self.results_times,
                               self.results,
                               self.ks_stats_samp,
                               self.err_max)

    def to_json(self):
        return dict(
            var_names=self.var_names,
            num_steps=self.num_steps,
            sampling_err_thresh=self.sampling_err_thresh,
            sig_figs=self.sig_figs,
            results_times=self.results_times.tolist(),
            results={n: v.tolist() for n, v in self.results.items()},
            ks_stats_samp=self.ks_stats_samp,
            err_max=self.err_max
        )

    @classmethod
    def from_json(cls, data):
        ks_stats_samp = data['ks_stats_samp']
        err_max = data['err_max']

        analysis = CuratorAnalysis(
            data['var_names'],
            int(data['num_steps']),
            float(data['sampling_err_thresh']),
            int(data['sig_figs']),
            np.asarray(data['results_times'], dtype=float),
            {n: np.asarray(v, dtype=float) for n, v in data['results'].items()},
            [float(d) for d in ks_stats_samp] if ks_stats_samp is not None else ks_stats_samp,
            {n: float(v) for n, v in err_max.items()} if err_max is not None else err_max
        )

        return analysis

    def __reduce__(self):
        return CuratorAnalysis.from_json, (self.to_json(),)

    def compare(self, modeler_metadata: sim_lib.Metadata):

        self.err_max = {n: 0.0 for n in self.var_names}
        for name in self.var_names:
            for i in range(self.num_steps):
                err_i = sim_lib.ecf_compare(
                    sim_lib.ecf(self.results[name][:modeler_metadata.sample_size, i],
                                sim_lib.get_eval_info_times(*modeler_metadata.ecf_eval_info[name][i])),
                    modeler_metadata.ecf_evals[name][i])
                if err_i > self.err_max[name]:
                    self.err_max[name] = err_i
        return self.err_max

    def pval(self, modeler_metadata: sim_lib.Metadata):
        
        self.compare(modeler_metadata)
        err_max = max(self.err_max.values())
        err_avg = np.average(self.ks_stats_samp)
        sample_size = len(self.ks_stats_samp)
        if err_max < err_avg:
            return 1.0
        q2 = (sample_size + 1) / sample_size * np.var(self.ks_stats_samp, ddof=1)
        lam2 = (err_max - err_avg) * (err_max - err_avg) / q2
        pr = np.floor((sample_size + 1) / sample_size * (
                    (sample_size - 1) / lam2 + 1)) / (sample_size + 1)
        return min(1.0, pr)


def generate_analysis(modeler_metadata: sim_lib.Metadata,
                      curator_res_dir: str,
                      sampling_err_thresh: float = 0.001):
    
    sample_times, results = get_curator_results(curator_res_dir, 
                                                list(modeler_metadata.ecf_evals.keys()), 
                                                2 * modeler_metadata.sample_size, 
                                                modeler_metadata.sample_times.shape[0], 
                                                modeler_metadata.sig_figs)
    analysis = CuratorAnalysis(list(modeler_metadata.ecf_evals.keys()),
                               modeler_metadata.sample_times.shape[0],
                               sampling_err_thresh,
                               modeler_metadata.sig_figs,
                               sample_times,
                               results)
    return analysis


def main(modeler_results_dir: str, curator_results_dir: str, output_dir: str = None):
    modeler_results_dir_6 = os.path.join(modeler_results_dir, 'modeler_6')
    modeler_results_dir_9 = os.path.join(modeler_results_dir, 'modeler_9')

    print('Fetching modeler data...', modeler_results_dir_6)
    modeler_metadata_6 = get_modeler_metadata(modeler_results_dir_6)
    print('Fetching modeler data...', modeler_results_dir_9)
    modeler_metadata_9 = get_modeler_metadata(modeler_results_dir_9)

    curator_results_dir_same = os.path.join(curator_results_dir, 'curator_results_same')
    curator_results_dir_diff = os.path.join(curator_results_dir, 'curator_results_diff')

    analyses = [
        ('sim_2_curator_pass', modeler_metadata_6, curator_results_dir_same),
        ('sim_2_curator_fail_sigfig', modeler_metadata_9, curator_results_dir_same),
        ('sim_2_curator_fail_params', modeler_metadata_6, curator_results_dir_diff)
    ]

    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    for name, modeler_metadata, curator_dir in analyses:
        print('Executing:', name, curator_dir)
        analysis = generate_analysis(modeler_metadata, curator_dir)
        analysis.compare(modeler_metadata)
        with open(os.path.join(output_dir, name + '.json'), 'w') as f:
            json.dump(dict(modeler=modeler_metadata.to_json(), curator=analysis.to_json()), f, indent=4)


if __name__ == '__main__':
    main(os.path.join(os.path.dirname(__file__), 'results', 'workflow_sim_2_py'),
         os.path.join(os.path.dirname(__file__), 'results', 'workflow_sim_2_py'),
         os.path.join(os.path.dirname(__file__), 'results', 'workflow_sim_2_py'))
