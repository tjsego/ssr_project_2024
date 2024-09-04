import numpy as np
from matplotlib import pyplot as plt
import stochastic_repro as sr
from stochastic_models import SBMLModel
from typing import Any, Dict, List, Optional, Tuple
import json


def plot_comp(_model, _ax, _sim_s, _row, _sim_d):
    for i, name in enumerate(_model.results_names):
        _sim_s.subplot_var(_ax[_row][i], name, color='gray', alpha=0.1)
        if _sim_d is not None:
            _sim_d.subplot_var(_ax[_row][i], name, linestyle='--')
        if _row == 0:
            _ax[_row][i].set_title(name)


class Test:

    def __init__(self, 
                 model: SBMLModel,
                 t_fin: float,
                 num_steps: int,
                 sample_times: List[float],
                 trials: List[int],
                 stochastic: bool = False,
                 num_var_steps: int = None,
                 num_var_pers: int = None):
        
        self.model = model
        self.t_fin = t_fin
        self.num_steps = num_steps
        self.sample_times = sample_times
        self.trials = trials
        self.stochastic = stochastic
        self.num_var_steps = num_var_steps
        self.num_var_pers = num_var_pers

        # Results

        self.sim_d: Optional[sr.SimSet] = None
        # Deterministic simulation

        self.sims_s: Optional[Dict[int, sr.SimSet]] = None
        # Stochastic simulations

        self.acc_diff_basic: Dict[float, Dict[str, Dict[int, float]]] = dict()
        # Basic convergence measurements

        self.acc_diff_kl_div: Optional[Dict[str, Dict[int, float]]] = None
        # Kullback-Leibler Divergence convergence measurements

        self.analysis_corr: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None
        # Correlation analysis results

        self.ecf: Optional[Dict[int, List[Dict[str, np.ndarray]]]] = None
        # Empirical characteristic functions for each trial and variable at each simulation time

        self.ecf_eval_info: Optional[Dict[int, List[Dict[str, sr.ECFEvalInfo]]]] = None
        # Empirical characteristic function evaluation information

        self.ecf_ks_stat: Optional[Dict[int, List[Dict[str, float]]]] = None
        # Empirical characteristic function evalution K-S statistics

        self.ecf_diff: Optional[Dict[int, Dict[str, float]]] = None
        # Maximum K-S statistic per variable per trial

        self.ecf_sampling: Optional[Dict[int, Tuple[float, float]]] = None
        # K-S self-similarity test statistics per trial

        self.ks_stats_sampling: Optional[Dict[int, List[float]]] = None
        # K-S self-similarity test values per trial

        self.ecf_diff_fits: Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = None

        self.means: Optional[Dict[int, Dict[str, np.ndarray]]] = None

        self.stdevs: Optional[Dict[int, Dict[str, np.ndarray]]] = None

    def to_json(self):
        json_data = dict(model=self.model.to_json(),
                         t_fin=self.t_fin,
                         num_steps=self.num_steps,
                         sample_times=self.sample_times,
                         trials=self.trials,
                         stochastic=self.stochastic)

        if self.num_var_steps is not None:
            json_data['num_var_steps'] = self.num_var_steps

        if self.num_var_pers is not None:
            json_data['num_var_pers'] = self.num_var_pers

        if self.sim_d is not None:
            json_data['sim_d'] = self.sim_d.to_json()

        if self.sims_s is not None:
            json_data['sims_s'] = {trial: sim.to_json() for trial, sim in self.sims_s.items()}

        if self.acc_diff_basic is not None:
            json_data['acc_diff_basic'] = self.acc_diff_basic

        if self.acc_diff_kl_div is not None:
            json_data['acc_diff_kl_div'] = self.acc_diff_kl_div

        if self.analysis_corr is not None:
            json_data['analysis_corr'] = {i: {s: (t[0].tolist(), t[1].tolist()) for s, t in d.items()}
                                          for i, d in self.analysis_corr.items()}

        if self.ecf is not None:
            json_data['ecf'] = {i: [{s: a.tolist() for s, a in d.items()} for d in v] for i, v in self.ecf.items()}
        
        if self.ecf_eval_info is not None:
            json_data['ecf_eval_info'] = self.ecf_eval_info
        
        if self.ecf_ks_stat is not None:
            json_data['ecf_ks_stat'] = self.ecf_ks_stat

        if self.ecf_diff is not None:
            json_data['ecf_diff'] = self.ecf_diff

        if self.ecf_sampling is not None:
            json_data['ecf_sampling'] = self.ecf_sampling

        if self.ks_stats_sampling is not None:
            json_data['ks_stats_sampling'] = self.ks_stats_sampling

        return json_data

    @staticmethod
    def from_json(json_data: dict):
        result = Test(model=SBMLModel.from_json(json_data['model']),
                      t_fin=float(json_data['t_fin']),
                      num_steps=int(json_data['num_steps']),
                      sample_times=[float(f) for f in json_data['sample_times']],
                      trials=[int(i) for i in json_data['trials']],
                      stochastic=bool(json_data['stochastic']))

        if 'num_var_steps' in json_data.keys():
            result.num_var_steps = int(json_data['num_var_steps'])

        if 'num_var_pers' in json_data.keys():
            result.num_var_pers = int(json_data['num_var_pers'])

        if 'sim_d' in json_data.keys():
            result.sim_d = sr.SimSet.from_json(json_data['sim_d'])
        
        if 'sims_s' in json_data.keys():
            result.sims_s = {int(name): sr.SimSet.from_json(sim) for name, sim in json_data['sims_s'].items()}
        
        if 'acc_diff_basic' in json_data.keys():
            result.acc_diff_basic = {float(f): {s: {int(i): float(ff) for i, ff in dd.items()} for s, dd in d.items()}
                                     for f, d in json_data['acc_diff_basic'].items()}
        
        if 'acc_diff_kl_div' in json_data.keys():
            result.acc_diff_kl_div = {s: {int(i): float(f) for i, f in d.items()}
                                      for s, d in json_data['acc_diff_kl_div'].items()}
        
        if 'analysis_corr' in json_data.keys():
            result.analysis_corr = {int(i): {s: (np.array(t[0]), np.array(t[1])) for s, t in d.items()}
                                    for i, d in json_data['analysis_corr'].items()}
        
        if 'ecf' in json_data.keys():
            result.ecf = {int(i): [{s: np.array(a) for s, a in d.items()} for d in v]
                          for i, v in json_data['ecf'].items()}
        
        if 'ecf_eval_info' in json_data.keys():
            result.ecf_eval_info = {int(i): [{s: (int(ei[0]), float(ei[1])) for s, ei in d.items()} for d in v]
                                    for i, v in json_data['ecf_eval_info'].items()}

        if 'ecf_ks_stat' in json_data.keys():
            result.ecf_ks_stat = {int(i): [{s: float(f) for s, f in d.items()} for d in v]
                                  for i, v in json_data['ecf_ks_stat'].items()}

        if 'ecf_diff' in json_data.keys():
            result.ecf_diff = {int(i): {s: float(f) for s, f in d.items()} for i, d in json_data['ecf_diff'].items()}

        if 'ecf_sampling' in json_data.keys():
            result.ecf_sampling = {int(i): (float(t[0]), float(t[1])) for i, t in json_data['ecf_sampling'].items()}

        if 'ks_stats_sampling' in json_data.keys():
            result.ks_stats_sampling = {int(i): [float(f) for f in d]
                                        for i, d in json_data['ks_stats_sampling'].items()}

        return result

    def clone(self):
        return Test.from_json(self.to_json())

    def save(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.to_json(), f)

    @staticmethod
    def load(file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)

        return Test.from_json(data)

    def execute_deterministic(self):
        if self.sim_d is not None:
            return
        self.sim_d = sr.SimSet(model=self.model, num_trials=1, stochastic=False, num_steps=self.num_steps,
                               t_fin=self.t_fin)
        self.sim_d.execute()

    def execute_stochastic(self, label=None):
        self.sims_s = {t: sr.SimSet(model=self.model,
                                    num_trials=t,
                                    stochastic=self.stochastic,
                                    num_steps=self.num_steps,
                                    t_fin=self.t_fin)
                       for t in self.trials}
        for t, sim in self.sims_s.items():
            if label is not None:
                label.value = f'Evaluating sample size {t}...'
            sim.execute_p()

    def measure_dist_diff_basic(self, filter: float = 0.0, progress_bar=None):
        self.acc_diff_basic[filter] = sr.measure_dist_diff(self.sims_s, self.sample_times, self.model.results_names,
                                                           self.trials, filter=filter, progress_bar=progress_bar)

    def measure_dist_div_kldiv(self, progress_bar=None):
        self.acc_diff_kl_div = sr.measure_dist_diff(self.sims_s, self.sample_times, self.model.results_names,
                                                    self.trials, comparator='kl_div', progress_bar=progress_bar)

    def measure_correlation(self):
        self.analysis_corr = sr.analysis_corr(self.sims_s, self.trials, self.model.results_names)

    def find_ecfs(self, num_workers: int = None, quiet=True):
        self.ecf, self.ecf_ks_stat, self.ecf_eval_info = sr.find_ecfs(
            self.sims_s, 
            self.model.results_names, 
            self.trials, 
            num_steps=self.num_var_steps,
            num_var_pers=self.num_var_pers,
            num_workers=num_workers, 
            quiet=quiet
        )

    def measure_ecf_diffs(self):
        self.ecf_diff = {trial: {n: max([ks_stat[n] for ks_stat in ks_data]) 
                                 for n in self.model.results_names} 
                         for trial, ks_data in self.ecf_ks_stat.items()}

    def test_sampling(self,
                      incr_sampling: int = None,
                      err_thresh: float = None,
                      max_sampling: int = None,
                      num_steps: int = None,
                      num_var_pers: int = None,
                      quiet: bool = None):
        kwargs = dict()
        if incr_sampling is not None:
            kwargs['incr_sampling'] = incr_sampling
        if err_thresh is not None:
            kwargs['err_thresh'] = err_thresh
        if max_sampling is not None:
            kwargs['max_sampling'] = max_sampling
        if num_steps is not None:
            kwargs['num_steps'] = num_steps
        if num_var_pers is not None:
            kwargs['num_var_pers'] = num_var_pers
        if quiet is not None:
            kwargs['quiet'] = quiet
        
        self.ecf_sampling = {}
        self.ks_stats_sampling = {}
        for t in self.trials:
            if quiet is not None and not quiet:
                print(f'Testing sample size {t}')
            self.ks_stats_sampling[t] = sr.test_sampling(self.sims_s[t].results, **kwargs)[0]
            self.ecf_sampling[t] = np.average(self.ks_stats_sampling[t]), np.std(self.ks_stats_sampling[t])

    @staticmethod
    def ecf_diff_fit_func(n, a, b):
        return a * n ** b

    def generate_ecf_diff_fits(self, **kwargs):
        if self.ecf_diff is None:
            raise RuntimeError

        data_p = []
        data_p_cov = []
        for i in range(2, len(self.trials) + 1):
            try:
                data_p_i, data_p_cov_i = sr.fit_ecf_diff(self.ecf_diff, self.model.results_names, self.trials[:i],
                                                         Test.ecf_diff_fit_func, **kwargs)
            except RuntimeError:
                data_p_i, data_p_cov_i = None, None
            data_p.append(data_p_i)
            data_p_cov.append(data_p_cov_i)
        self.ecf_diff_fits = data_p, data_p_cov

    def generate_ecf_sampling_fits(self, **kwargs):
        if self.ecf_sampling is None:
            raise RuntimeError

        data_p = []
        data_p_cov = []
        for i in range(2, len(self.trials) + 1):
            x_data = self.trials[:i]
            y_data = [self.ecf_sampling[t][0] for t in x_data]
            err_data = [self.ecf_sampling[t][1] for t in x_data]
            try:
                data_p_i, data_p_cov_i = sr.fit_data(Test.ecf_diff_fit_func, x_data, y_data, sigma=err_data, **kwargs)
            except RuntimeError:
                data_p_i, data_p_cov_i = None, None
            data_p.append(data_p_i)
            data_p_cov.append(data_p_cov_i)

        self.ecf_sampling_fits = data_p, data_p_cov

    def measure_stats(self):
        if self.sims_s is None:
            raise RuntimeError
        
        self.means, self.stdevs = sr.measure_stats(self.sims_s)

    def run(self, quiet=True):
        
        new_pool = sr.get_pool() is None
        if new_pool:
            sr.start_pool()

        self.execute_deterministic()
        self.execute_stochastic()
        # self.measure_dist_diff_basic()
        # self.measure_dist_diff_basic(0.5)
        # self.measure_dist_div_kldiv()
        # self.measure_correlation()
        self.find_ecfs(quiet=quiet)
        self.measure_ecf_diffs()
        self.test_sampling(err_thresh=1E-3, quiet=quiet)
        
        if new_pool:
            sr.close_pool()

    def max_ks_stat_time(self, _trial: int):
        ks_stat_max, ks_stat_max_idx = -1, -1
        for i, ks_data in enumerate(self.ecf_ks_stat[_trial]):
            ks_data_max = max(ks_data.values())
            if ks_data_max > ks_stat_max:
                ks_stat_max = ks_data_max
                ks_stat_max_idx = i
        return self.sims_s[_trial].results_time[ks_stat_max_idx]

    def min_final_eval_time(self, _trial: int):
        t_fin_min, t_fin_min_idx = None, -1
        for i, ei in enumerate(self.ecf_eval_info[_trial]):
            ei_min = min([v[1] for v in ei.values()])
            if t_fin_min is None or ei_min < t_fin_min:
                t_fin_min = ei_min
                t_fin_min_idx = i
        return self.sims_s[_trial].results_time[t_fin_min_idx]

    def plot_results_deterministic(self):
        if self.sim_d is None:
            raise RuntimeError

        fig, ax = plt.subplots(1, len(self.model.results_names),
                               figsize=(12.0, 2.0), layout='compressed', squeeze=False)
        ax = ax[0]
        for i, name in enumerate(self.model.results_names):
            self.sim_d.subplot_var(ax[i], name)
            ax[i].set_title(name)
            ax[i].set_xlabel('Time')
        return fig, ax

    def plot_results_stochastic(self, plot_det=True):
        if self.sims_s is None:
            raise RuntimeError

        fig, axs = plt.subplots(len(self.trials), len(self.model.results_names), sharex=True,
                                figsize=(3 * len(self.model.results_names), 2 * len(self.trials)),
                                layout='compressed', squeeze=False)

        for i, t in enumerate(self.trials):
            plot_comp(self.model, axs, self.sims_s[t], i, self.sim_d if plot_det else None)
            axs[i][0].set_ylabel(f'Sample size {t}')
        for j in range(len(self.model.results_names)):
            axs[-1][j].set_xlabel('Time')
        return fig, axs

    def plot_distributions(self):
        if self.sims_s is None:
            raise RuntimeError

        fig, axs = plt.subplots(len(self.trials), len(self.model.results_names),
                                figsize=(3 * len(self.model.results_names), 2.0 * len(self.trials)),
                                sharex=True, layout='compressed', squeeze=False)
        for j, trial in enumerate(self.trials):
            for i, name in enumerate(self.model.results_names):

                flat_time = []
                flat_res = []

                for idx, sample_time in enumerate(self.sims_s[trial].time):

                    res = self.sims_s[trial].extract_var_index(name, idx)
                    flat_time.extend([sample_time] * len(res))
                    flat_res.extend(res)

                axs[j][i].hist2d(flat_time, flat_res, density=True, bins=(len(self.sims_s[self.trials[0]].time), 10))
            axs[j][0].set_ylabel(f'Sample size {trial}')
        for i, name in enumerate(self.model.results_names):
            axs[0][i].set_title(name)
            axs[-1][i].set_xlabel('Time')
        return fig, axs

    def plot_distributions_compare(self, trial: int, num_bins: int = 10):
        if self.sims_s is None:
            raise RuntimeError
        elif trial not in self.trials:
            raise ValueError(f'Input {trial} not available ({self.trials})')

        fig, ax = plt.subplots(2, len(self.model.results_names), figsize=(12.0, 4.0),
                               sharex=True, layout='compressed', squeeze=False)

        for i, name in enumerate(self.model.results_names):

            flat_time1 = []
            flat_res1 = []
            flat_time2 = []
            flat_res2 = []

            for idx, sample_time in enumerate(self.sims_s[trial].time):

                res = self.sims_s[trial].extract_var_index(name, idx)
                n = int(len(res) / 2)
                res1 = res[:n]
                res2 = res[n:]
                flat_time1.extend([sample_time] * len(res1))
                flat_res1.extend(res1)
                flat_time2.extend([sample_time] * len(res2))
                flat_res2.extend(res2)

            ax[0][i].hist2d(flat_time1, flat_res1, density=True, bins=(len(self.sims_s[trial].time), num_bins))
            ax[1][i].hist2d(flat_time2, flat_res2, density=True, bins=(len(self.sims_s[trial].time), num_bins))
        ax[0][0].set_ylabel(f'Sample size {trial}: Set 1')
        ax[1][0].set_ylabel(f'Sample size {trial}: Set 2')
        for i, name in enumerate(self.model.results_names):
            ax[0][i].set_title(name)
            ax[-1][i].set_xlabel('Time')
        return fig, ax

    def plot_dist_diff(self, acc_diff: Dict[str, Dict[int, float]]):

        fig, ax = plt.subplots(1, len(self.model.results_names), sharey=True,
                               figsize=(12.0, 2.0), layout='compressed', squeeze=False)
        ax = ax[0]

        for i, name in enumerate(self.model.results_names):
            ax[i].scatter(acc_diff[name].keys(), acc_diff[name].values())
            ax[i].set_title(name)
            ax[i].set_xscale('log')
            ax[i].set_xlabel('Sample size')
            ax[i].set_ylim(-0.05, max(1.0, 1.05 * max(acc_diff[name].values())))

        max_val = -1.0
        for name in self.model.results_names:
            max_val = max(max_val, max(acc_diff[name].values()))
        ax[0].set_ylim(-0.05, 1.05 * min(1, max_val))
        
        return fig, ax

    def plot_correlations(self):
        if self.analysis_corr is None:
            raise RuntimeError

        fig1, ax1 = plt.subplots(len(self.trials), len(self.model.results_names),
                                 figsize=(12.0, 2.0 * len(self.trials)), layout='compressed', squeeze=False)
        fig2, ax2 = plt.subplots(len(self.trials), len(self.model.results_names),
                                 figsize=(12.0, 2.0 * len(self.trials)), layout='compressed', squeeze=False)

        for j, trial in enumerate(self.trials):
            for i, name in enumerate(self.model.results_names):
                corr, corr_max = self.analysis_corr[trial][name]

                ax1[j][i].imshow(corr, vmin=-1, vmax=1)
                ax2[j][i].plot(corr_max)
                ax2[j][i].set_ylim(-1.1, 1.1)
            ax1[j][0].set_ylabel(f'Sample size {trial}')
            ax2[j][0].set_ylabel(f'Sample size {trial}')
        for i, name in enumerate(self.model.results_names):
            ax1[0][i].set_title(name)
            ax2[0][i].set_title(name)

        return fig1, ax1, fig2, ax2

    def plot_ecf(self, time: float, eval_info: Dict[str, sr.ECFEvalInfo] = None, fig=None):
        if self.ecf is None:
            raise RuntimeError
        
        eval_times_override = None
        if eval_info is not None:
            eval_times_override = {n: sr.get_eval_info_times(eval_info[n]) for n in eval_info.keys()}

        args = 2, len(self.model.results_names)
        if fig is None:
            kwargs = dict(sharex=False, sharey=False, figsize=(12.0, 2.0 * 2), layout='compressed', squeeze=False)
            fig, ax = plt.subplots(*args, **kwargs)
        else:
            ax = fig.subplots(*args, squeeze=False)

        for trial in self.trials:
            idx = self.sims_s[trial].get_time_index(time)
            for j, name in enumerate(self.model.results_names):
                ecfs = self.ecf[trial][idx][name]
                if eval_times_override is None:
                    eval_times = sr.get_eval_info_times((ecfs.shape[0], self.ecf_eval_info[trial][idx][name][1]))
                    # Handle erroneous rounding issue
                    if eval_times.shape[0] > ecfs.shape[0]:
                        eval_times = eval_times[:ecfs.shape[0]]
                else:
                    eval_times = eval_times_override[name]
                ax[0][j].plot(eval_times, ecfs[:, 0], label=f'Trials: {trial}')
                ax[1][j].plot(eval_times, ecfs[:, 1], label=f'Trials: {trial}')

        for j, name in enumerate(self.model.results_names):
            ax[0][j].set_title(name)
        ax[0][0].set_ylabel('Real')
        ax[1][0].set_ylabel('Imaginary')
        fig.suptitle(f'Empirical characteristic functions (time={time})')
        fig.legend(labels=[f'Trials: {trial}' for trial in self.trials])

        return fig, ax

    def plot_ecf_diffs(self, fig=None):
        if self.ecf_diff is None:
            raise RuntimeError

        args = 1, len(self.model.results_names)
        if fig is None:
            kwargs = dict(sharey=False, figsize=(12.0, 4.0), layout='compressed', squeeze=False)
            fig, ax = plt.subplots(*args, **kwargs)
        else:
            ax = fig.subplots(*args, squeeze=False)
        ax = ax[0]
        for i, name in enumerate(self.model.results_names):
            ax[i].scatter(self.trials, [self.ecf_diff[trial][name] for trial in self.trials])
            ax[i].set_xlabel('Sample size')
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
            ax[i].set_title(name)
        ax[0].set_ylabel('EFECT error')
        fig.suptitle('Measure of difference between empirical characteristic functions')

        return fig, ax

    def plot_ecf_sampling(self, fig=None):
        if self.ecf_sampling is None:
            raise RuntimeError

        args = 1, 1
        if fig is None:
            kwargs = dict(figsize=(6.0, 4.0), layout='compressed')
            fig, ax = plt.subplots(*args, **kwargs)
        else:
            ax = fig.subplots(*args)
        
        avg = np.asarray([self.ecf_sampling[t][0] for t in self.trials], dtype=float)
        std = np.asarray([self.ecf_sampling[t][1] for t in self.trials], dtype=float)
        ax.errorbar(self.trials, avg, yerr=std, marker='o', linestyle='none')
        ax.set_xlabel('Sample size')
        ax.set_ylabel('EFECT error')
        ax.set_xscale('log')
        ax.set_yscale('log')

        fig.suptitle('Test for reproducibility EFECT error')

        return fig, ax

    def plot_ecf_comparison(self, time: float):
        fig_r, ax_r = plt.subplots(len(self.trials), len(self.model.results_names), sharex=False, sharey=False,
                                   figsize=(12.0, 2.0 * len(self.trials)), layout='compressed', squeeze=False)
        fig_i, ax_i = plt.subplots(len(self.trials), len(self.model.results_names), sharex=False, sharey=False,
                                   figsize=(12.0, 2.0 * len(self.trials)), layout='compressed', squeeze=False)

        for i, trial in enumerate(self.trials):
            n = int(trial / 2)
            time_idx = self.sims_s[self.trials[0]].get_time_index(time)
            eval_info_trial = self.ecf_eval_info[trial][time_idx]

            for j, name in enumerate(self.model.results_names):
                eval_t = sr.get_eval_info_times(eval_info_trial[name])
                res = self.sims_s[trial].extract_var_index(name, time_idx)

                ecf1 = sr.ecf(res[n:], eval_t)
                ecf2 = sr.ecf(res[:n], eval_t)

                ax_r[i][j].plot(eval_t, ecf1[:, 0])
                ax_r[i][j].plot(eval_t, ecf2[:, 0])
                ax_i[i][j].plot(eval_t, ecf1[:, 1])
                ax_i[i][j].plot(eval_t, ecf2[:, 1])

            ax_r[i][0].set_ylabel(f'Trials: {trial}')
            ax_i[i][0].set_ylabel(f'Trials: {trial}')
        for j, name in enumerate(self.model.results_names):
            ax_r[0][j].set_title(name)
            ax_i[0][j].set_title(name)
            ax_r[-1][j].set_xlabel('Time')
            ax_i[-1][j].set_xlabel('Time')
        fig_r.suptitle(f'Empirical characteristic functions (time={time}, real)')
        fig_i.suptitle(f'Empirical characteristic functions (time={time}, imaginary)')

        return fig_r, ax_r, fig_i, ax_i

    def plot_ecf_diff_fits(self, fig_axs=None):
        if self.ecf_diff_fits is None:
            raise RuntimeError

        if fig_axs is not None:
            fig, axs = fig_axs
        else:
            fig, axs = plt.subplot(1, len(self.model.results_names), sharey=False,
                                   figsize=(12.0, 4.0), layout='compressed', squeeze=False)
            axs = axs[0]

        for i, name in enumerate(self.model.results_names):
            ax = axs[i]

            for j, data_p_i in enumerate(self.ecf_diff_fits[0]):
                if data_p_i is not None:
                    model_p = data_p_i[name]
                    ax.plot(self.trials, [Test.ecf_diff_fit_func(n, *model_p) for n in self.trials],
                            label=f'Sample size {self.trials[j+1]}')
        return fig, axs

    def plot_ecf_sampling_fits(self, fig_axs=None):
        if self.ecf_sampling_fits is None:
            raise RuntimeError

        if fig_axs is not None:
            fig, axs = fig_axs
        else:
            fig, axs = plt.subplot(1, 1, figsize=(6.0, 4.0), layout='compressed', squeeze=False)

        for i, data_f in enumerate(self.ecf_sampling_fits[0]):
            if data_f is not None:
                axs.plot(self.trials, [Test.ecf_diff_fit_func(n, *data_f) for n in self.trials],
                         label=f'Sample size {self.trials[i+1]}')
        return fig, axs

    def plot_ks_sampling(self):
        if self.ks_stats_sampling is None:
            raise RuntimeError

        fig, axs = plt.subplots(len(self.trials), 1, sharex=True, figsize=(12.0, 3.0 * len(self.trials)),
                                layout='compressed', squeeze=False)
        axs = axs[:][0]
        for i, t in enumerate(self.trials):
            ax = axs[i]
            ax.hist(self.ks_stats_sampling[t], density=True)
            ax.set_ylabel(f'Sample size {t}')
        axs[-1][0].set_xlabel('EFECT error')
        fig.suptitle('EFECT error density plots')
        return fig, axs

    def plot_stats(self, fig_axs=None):
        if self.means is None:
            self.measure_stats()

        if fig_axs is not None:
            fig, axs = fig_axs
        else:
            fig, axs = plt.subplots(1, len(self.model.results_names), sharey=False, figsize=(12.0, 4.0),
                                    layout='compressed', squeeze=False)
            axs = axs[0]

        colors = {name: dict() for name in self.model.results_names}

        for i, name in enumerate(self.model.results_names):
            ax = axs[i]

            for trial in self.trials:

                lns = ax.plot(self.sims_s[trial].results_time, self.means[trial][name], label=f'Samples {trial}')
                colors[name][trial] = lns[0].get_color()

            ax.set_title(name)

        fig.legend([f'Samples {trial}' for trial in self.trials])

        for i, name in enumerate(self.model.results_names):
            ax = axs[i]

            for trial in self.trials:
                times = self.sims_s[trial].results_time
                means = self.means[trial][name]
                stdevs = self.stdevs[trial][name]
                color = colors[name][trial]
                ax.plot(times, means - stdevs, color=color, linestyle='--')
                ax.plot(times, means + stdevs, color=color, linestyle='--')

            ax.set_title(name)
        
        return fig, axs
