import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import antimony
from roadrunner import RoadRunner
from random import randint
import os
import sys
import scipy
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Tuple
from stochastic_models import SBMLModel
import json

# Numba seems to only behave well on Windows
if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
    has_numba = False
else:
    import numba
    has_numba = True


lib_pool: Optional[mp.Pool] = None


def _seed_pool(*args, **kwargs):
    seed = os.getpid()
    np.random.seed(seed)
    return seed


def start_pool(num_workers: int = None):
    if num_workers is None:
        num_workers = mp.cpu_count()

    global lib_pool
    lib_pool = mp.Pool(num_workers)
    results = lib_pool.map(_seed_pool, [tuple()] * num_workers)
    return results


def get_pool():
    return lib_pool


def close_pool():
    global lib_pool
    lib_pool = None


def antimony_to_sbml(model_string: str) -> str:
    antimony.clearPreviousLoads()
    antimony.loadAntimonyString(model_string)
    module_name = antimony.getMainModuleName()
    return antimony.getSBMLString(module_name)


def generate_dists(params_info: dict):
    param_dists = {}

    def _func(_dist_name, *args):
        f = getattr(scipy.stats, _dist_name)

        def _impl():
            return f.rvs(*args, random_state=np.random.default_rng())
        return _impl
    for k, v in params_info.items():
        dist_name, args = v
        param_dists[k] = _func(dist_name, *args)
    return param_dists


def apply_dists(rr, param_dists):
    for k, v in param_dists.items():
        rr[k] = v()


def apply_mods(rr, mods):
    for k, v in mods.items():
        rr[k] = v


def _exec_rr(rr: RoadRunner, t_fin: float, num_steps: int, stochastic: bool, mods, param_dists):
    while True:
        rr.resetAll()
        if stochastic:
            rr.integrator.integrator = 'gillespie'
            rr.integrator.seed = randint(0, int(1E6))
        if mods is not None:
            apply_mods(rr, mods)
        if param_dists is not None:
            apply_dists(rr, generate_dists(param_dists))
        try:
            return rr.simulate(0, t_fin, num_steps)
        except RuntimeError:
            pass


def ecf(var_vals: np.ndarray, func_evals: np.ndarray):
    """
    Empirical characteristic function

    :param var_vals: trajectory values
    :param func_evals: independent variable values at which to compute the empirical characteristic function
    :return: empirical characteristic function evaluations; first dim is evaluations; second dim is real and imaginary components
    :rtype: np.ndarray
    """
    result = np.zeros((func_evals.shape[0], 2))

    func_evals_mat = np.repeat(func_evals[:, np.newaxis], var_vals.shape[0], 1)
    var_vals_mat = np.repeat(var_vals[:, np.newaxis], func_evals.shape[0], 1).T
    x = func_evals_mat * var_vals_mat
    result[:, 0] = np.average(np.cos(x), 1)
    result[:, 1] = np.average(np.sin(x), 1)

    return result


def _ecf_njit(var_vals: np.ndarray, func_evals: np.ndarray):
    """
    Empirical characteristic function

    :param var_vals: trajectory values
    :param func_evals: independent variable values at which to compute the empirical characteristic function
    :return: empirical characteristic function evaluations; first dim is evaluations; second dim is real and imaginary components
    :rtype: np.ndarray
    """
    result = np.zeros((func_evals.shape[0], 2))

    for i in range(func_evals.shape[0]):
        t = func_evals[i]
        result[i, 0] = np.average(np.cos(var_vals * t))
        result[i, 1] = np.average(np.sin(var_vals * t))

    return result


if has_numba:
    ecf = numba.njit(_ecf_njit)


def ecf_mag(var_vals: np.ndarray):
    """
    Magnitude of an empirical characteristic function

    var_vals: empirical characteristic function evaluations; first dim is evaluations; second dim is real and imaginary components
    """
    return np.sqrt(np.multiply(var_vals[:, 0], var_vals[:, 0]) + np.multiply(var_vals[:, 1], var_vals[:, 1]))


if has_numba:
    ecf_mag = numba.njit(ecf_mag)


def ecf_compare(res_1_r: np.ndarray, res_1_i: np.ndarray, res_2_r: np.ndarray, res_2_i: np.ndarray):
    """
    Compare empirical chacteristic function values using the Kolmogorov-Smirnov statistic
    """
    return np.max(np.sqrt(np.square(res_1_r - res_2_r) + np.square(res_1_i - res_2_i)))


if has_numba:
    ecf_compare = numba.njit(ecf_compare)


ECFEvalInfo = Tuple[int, float]
"""
Evaluation information for an empirical characteristic function: 
number of evaluations and final value of independent variable.
"""

DEF_EVAL_FIN = 1.0
DEF_EVAL_NUM = 100
DEF_KS_CONV = 0.05
DEF_EVAL_NUM_ITER = 2.0
DEF_NUM_VAR_PERS = 5


def get_eval_info_times(ecf_eval_info: ECFEvalInfo):
    return np.linspace(0.0, ecf_eval_info[1], ecf_eval_info[0])


if has_numba:
    get_eval_info_times = numba.njit(get_eval_info_times)


class ECF:

    def __init__(self, 
                 results: Dict[str, np.ndarray],
                 num_steps: int = DEF_EVAL_NUM,
                 num_var_pers: int = DEF_NUM_VAR_PERS):
        
        self.num_steps = {n: num_steps for n in results.keys()}
        self.results = results

        self.incr_max = {}
        self.ks_stat = {}

        for name, res in results.items():
            if np.std(res) == 0.0:
                self.incr_max[name] = 1 / self.num_steps[name]
                self.ks_stat[name] = 0.0
                continue

            self.incr_max[name] = 2 * num_var_pers * np.pi / np.std(results[name]) / self.num_steps[name]

            eval_pts = self.eval_pts_name(name)
            n = int(res.shape[0] / 2)
            ecf1 = ecf(res[:n], eval_pts)
            ecf2 = ecf(res[n:], eval_pts)
            self.ks_stat[name] = ecf_compare(ecf1[:, 0], ecf1[:, 1], ecf2[:, 0], ecf2[:, 1])

    @property
    def results_names(self) -> List[str]:
        return list(self.results.keys())

    def eval_pts_name(self, name: str, num_steps: int = None, incr: float = None) -> np.ndarray:
        if num_steps is None:
            num_steps = self.num_steps[name]
        if incr is None:
            incr = self.incr_max[name]
        
        try:
            return np.arange(0.0, (num_steps+1) * incr, incr, dtype=float)
        except ValueError as e:
            print('name =', name)
            print('num_steps =', num_steps)
            print('incr =', incr)
            raise e

    def eval_pts(self,
                 num_steps: Dict[str, Optional[int]] = None,
                 incr: Dict[str, Optional[float]] = None) -> Dict[str, np.ndarray]:
        if num_steps is None:
            num_steps = self.num_steps
        else:
            for k, v in self.num_steps.items():
                if k not in num_steps.keys():
                    num_steps[k] = v
        if incr is None:
            incr = self.incr_max
        else:
            for k, v in self.incr_max.items():
                if k not in incr.keys():
                    incr[k] = v

        return {k: self.eval_pts_name(k, v, incr[k]) for k, v in self.num_steps.items()}

    def __call__(self, t: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
        if t is None:
            t = self.eval_pts
        return {n: ecf(self.results[n], t[n]) for n in self.results_names}


def ecf_sample(_results: Dict[str, np.ndarray], 
               num_steps=DEF_EVAL_NUM,
               num_var_pers: int = DEF_NUM_VAR_PERS):
    ecf_test = ECF(_results, num_steps, num_var_pers)

    eval_fin = {n: ecf_test.num_steps[n] * ecf_test.incr_max[n] for n in ecf_test.results_names}

    return ecf_test.ks_stat, eval_fin, ecf_test.num_steps


class SimSet:
    
    def __init__(self, 
                 model: SBMLModel,
                 num_trials: int, 
                 stochastic: bool = True,
                 num_steps: int = 100, 
                 t_fin: float = 100.0):
        
        self.model = model
        self.num_trials = num_trials
        self.stochastic = stochastic
        self.num_steps = num_steps
        self.t_fin = t_fin
        
        self.results = None
        self.results_time = None
        self.progress_bar = None

    def to_json(self) -> dict:
        json_data = dict(model=self.model.to_json(), 
                         num_trials=self.num_trials,
                         stochastic=self.stochastic,
                         num_steps=self.num_steps,
                         t_fin=self.t_fin)
        if self.results is not None:
            json_data['results'] = {name: results.tolist() for name, results in self.results.items()}
        if self.results_time is not None:
            json_data['results_time'] = self.results_time.tolist()
        return json_data

    @staticmethod
    def from_json(json_data: dict):
        result = SimSet(model=SBMLModel.from_json(json_data['model']),
                        num_trials=int(json_data['num_trials']),
                        stochastic=bool(json_data['stochastic']),
                        num_steps=int(json_data['num_steps']),
                        t_fin=float(json_data['t_fin']))

        if 'results' in json_data.keys():
            result.results = {name: np.array(result) for name, result in json_data['results'].items()}
        if 'results_time' in json_data.keys():
            result.results_time = np.array([float(f) for f in json_data['results_time']])

        return result

    def save(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.to_json(), f)

    @staticmethod
    def load(file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)

        return SimSet.from_json(data)

    def make_rr(self):
        rr = RoadRunner(self.model.sbml)
        if self.stochastic:
            rr.integrator = 'gillespie'
            rr.integrator.seed = randint(0, int(1E6))
        return rr

    @staticmethod
    def convert_rr_results(res) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        out_time = np.asarray(res[0]['time'])

        results_names = [s.replace('[', '').replace(']', '') for s in res[0].colnames if s != 'time']

        dim = len(res), len(out_time)
        out_results = {name: np.zeros(dim) for name in results_names}
        for name in results_names:
            for i, r in enumerate(res):
                out_results[name][i, :] = r[f'[{name}]']
        return out_time, out_results

    def execute(self):
        if self.progress_bar is not None:
            self.progress_bar.min = 0
            self.progress_bar.max = self.num_trials
            self.progress_bar.value = 0
        
        rrs = [self.make_rr() for _ in range(self.num_trials)]
        if self.model.mods is not None:
            [apply_mods(rr, self.model.mods) for rr in rrs]
        if self.model.param_dists is not None:
            param_dists = generate_dists(self.model.param_dists)
            [apply_dists(rr, param_dists) for rr in rrs]

        results = []
        for rr in rrs:
            results.append(rr.simulate(0, self.t_fin, self.num_steps))

            if self.progress_bar is not None:
                self.progress_bar.value += 1

        self.results_time, self.results = SimSet.convert_rr_results(results)

    def execute_p(self, num_workers: int = None):
        if num_workers is None:
            num_workers = mp.cpu_count()
        num_workers = min(num_workers, self.num_trials)

        args = (self.make_rr(), self.t_fin, self.num_steps, self.stochastic, self.model.mods, self.model.param_dists)
        pool = get_pool()
        if pool is None:
            pool = mp.Pool(num_workers)
        results = pool.starmap(_exec_rr, [args for _ in range(self.num_trials)])
        self.results_time, self.results = SimSet.convert_rr_results(results)

    def plot_var(self, name: str, **kwargs):
        if self.results is None:
            raise RuntimeError
        res = self.results[name]
        for i in range(res.shape[0]):
            plt.plot(self.results_time, res[i, :], **kwargs)

    def plot_varx(self, xname: str, yname: str, **kwargs):
        if self.results is None:
            raise RuntimeError
        res_x = self.results[xname]
        res_y = self.results[yname]
        for i in range(res_x.shape[0]):
            plt.plot(res_x[i, :], res_y[i, :], **kwargs)

    def subplot_var(self, ax, name: str, **kwargs):
        if self.results is None:
            raise RuntimeError
        res = self.results[name]
        for i in range(res.shape[0]):
            ax.plot(self.results_time, res[i, :], **kwargs)

    def subplot_varx(self, ax, xname: str, yname: str, **kwargs):
        if self.results is None:
            raise RuntimeError
        res_x = self.results[xname]
        res_y = self.results[yname]
        for i in range(res_x.shape[0]):
            ax.plot(res_x[i, :], res_y[i, :], **kwargs)

    def subplot_varxt(self, ax, xname: str, yname: str, **kwargs):
        if self.results is None:
            raise RuntimeError
        sim_time = np.asarray(self.time)
        sim_time /= max(sim_time)
        sim_timen = np.ones_like(sim_time) - sim_time
        res_x = self.results[xname]
        res_y = self.results[yname]
        for i in range(res_x.shape[0]):
            res_xi = res_x[i, :]
            res_yi = res_y[i, :]
            for i in range(len(sim_time) - 1):
                ax.plot(res_xi[i:i+2], res_yi[i:i+2], color=(0.0, sim_time[i], sim_timen[i]), **kwargs)

    @property
    def time(self):
        if self.results_time is None:
            raise AttributeError
        return self.results_time

    @property
    def results_names(self):
        if self.results is None:
            raise AttributeError
        return list(self.results.keys())

    def get_name_index(self, name: str):
        return self.results_names.index(name)

    def get_time_index(self, time: float):
        dist = 1E6
        idx = -1
        for i, t in enumerate(self.time):
            d = abs(t - time)
            if d < dist:
                idx = i
                dist = d
            elif idx >= 0:
                return idx
        return idx

    def extract_var_index(self, name: str, idx: int) -> np.ndarray:
        return self.results[name].T[idx, :]

    def extract_var_time(self, name: str, time: float) -> np.ndarray:
        return self.results[name].T[self.get_time_index(time), :]

    def min_index(self, name: str, idx: int):
        return min(self.extract_var_index(name, idx))

    def min_time(self, name: str, time: float):
        return min(self.extract_var_time(name, time))
    
    def max_index(self, name: str, idx: int):
        return max(self.extract_var_index(name, idx))

    def max_time(self, name: str, time: float):
        return max(self.extract_var_time(name, time))

    def extract_stats(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        r_med = np.zeros((self.num_steps,))
        r_std = np.zeros((self.num_steps,))
        for idx in range(self.num_steps):
            r = self.extract_var_index(name, idx)
            r_med[idx] = np.average(r)
            r_std[idx] = np.std(r)

        return r_med, r_std

    def med(self, name: str) -> np.ndarray:
        return self.extract_stats(name)[0]

    def std(self, name: str) -> np.ndarray:
        return self.extract_stats(name)[1]

    def range_stats(self, name: str, _fact: int = 1, allow_neg: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        center, std = self.extract_stats(name)
        if allow_neg:
            f = lambda x: float(x)
        else:
            f = lambda x: max(0.0, float(x))
        
        dim = (len(center),)
        lower = np.zeros(dim)
        upper = np.zeros(dim)
        for i in range(dim[0]):
            lower[i] = f(center[i] - std[i] * _fact)
            upper[i] = f(center[i] + std[i] * _fact)
        return lower, upper

    def plot_range(self, name: str, fact=1, alpha=0.5, allow_neg: bool = False):
        lower, upper = self.range_stats(name, fact, allow_neg)
        plt.fill_between(self.time, lower, upper, color='gray', alpha=alpha)

    def subplot_range(self, ax, name: str, fact=1, alpha=0.5, allow_neg: bool = False):
        lower, upper = self.range_stats(name, fact, allow_neg)
        ax.fill_between(self.time, lower, upper, color='gray', alpha=alpha)


def _results_set_min_max(results: np.ndarray, idx: int):
    r = results.T[idx, :].T
    return min(r), max(r)


if has_numba:
    _results_set_min_max = numba.njit(_results_set_min_max)


def sim_set_min_max(sims: List[SimSet], name: str, time: float = None, idx: int = None):
    if time is None and idx is None:
        raise RuntimeError('Must specify a time or index')

    if idx is None:
        idx = sims[0].get_time_index(time)

    dim = (len(sims),)
    min_r = np.zeros(dim)
    max_r = np.zeros(dim)
    for i, sim in enumerate(sims):
        min_r[i], max_r[i] = _results_set_min_max(sim.results[name], idx)

    return min(min_r), max(max_r)


def sampled_hist(sims, sample_times, results_names, trials, num_workers: int = None, progress_bar=None):

    if num_workers is None:
        num_workers = mp.cpu_count() * 2

    sampled_min = {}
    sampled_max = {}

    num_jobs = len(sample_times) * len(results_names) * len(trials)
    if progress_bar is not None:
        progress_bar.max = len(sample_times) + num_jobs
        progress_bar.value = 0

    for sample_time in sample_times:

        sampled_min[sample_time] = {}
        sampled_max[sample_time] = {}

        for name in results_names:
            sampled_min[sample_time][name], sampled_max[sample_time][name] = sim_set_min_max(list(sims.values()),
                                                                                             name,
                                                                                             time=sample_time)

        if progress_bar is not None:
            progress_bar.value += 1

    result = {}
    for sample_time in sample_times:
        result[sample_time] = {}
        for name in results_names:
            result[sample_time][name] = {}

    for sample_time in sample_times:
        for name in results_names:
            for trial in trials:
                data = sims[trial].extract_var_time(name, sample_time)
                result[sample_time][name][trial] = np.histogram(data, 
                                                                density=True, 
                                                                range=(sampled_min[sample_time][name],
                                                                       sampled_max[sample_time][name]))
                if progress_bar is not None:
                    progress_bar.value += 1

    return result


def _measure_dist_diff(data, name, trial, sampled_min, sampled_max, filter):
    n = int(len(data) / 2)
    data1 = data[:n]
    data2 = data[n:]
    hist1, bins = np.histogram(data1, density=True, range=(sampled_min, sampled_max))
    hist2 = np.histogram(data2, density=True, range=(sampled_min, sampled_max))[0]

    den = np.minimum(hist1, hist2)
    res_filter = den * (bins[1:] - bins[:-1]) > filter

    num_filtered = np.absolute(hist1[res_filter] - hist2[res_filter])
    result = (num_filtered / den[res_filter])
    
    return name, trial, np.average(result), result.shape[0]


def _measure_dist_diff_kullback_leibler_divergence(data, name, trial, sampled_min, sampled_max, filter):
    n = int(len(data) / 2)
    data1 = data[:n]
    data2 = data[n:]
    hist1, bins = np.histogram(data1, density=True, range=(sampled_min, sampled_max))
    hist2 = np.histogram(data2, density=True, range=(sampled_min, sampled_max))[0]

    dbins = bins[1:] - bins[:-1]
    mask = dbins * np.minimum(hist1, hist2) > filter
    dbins_mask = dbins[mask]
    hist1_mask = hist1[mask] * dbins_mask
    hist2_mask = hist2[mask] * dbins_mask

    result = np.multiply(hist1_mask, np.abs(np.log(np.divide(hist1_mask, hist2_mask))))
    
    return name, trial, np.average(result), result.shape[0]


comparators = {
    'abs_rel_diff': _measure_dist_diff,
    'kl_div': _measure_dist_diff_kullback_leibler_divergence
}


def measure_dist_diff(sims, 
                      sample_times, 
                      results_names, 
                      trials, 
                      filter: float = 0.0, 
                      comparator='abs_rel_diff',
                      num_workers: int = None, 
                      progress_bar = None):

    if comparator not in comparators.keys():
        raise ValueError(f'Invalid comparator {comparator}. Valid comparators are {list(comparators.keys())}.')
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(results_names) * len(trials))

    sampled_min = {}
    sampled_max = {}

    num_jobs = len(sample_times) * len(results_names) * len(trials)
    if progress_bar is not None:
        progress_bar.max = len(sample_times) + num_jobs
        progress_bar.value = 0

    sample_indices = [sims[trials[0]].get_time_index(sample_time) for sample_time in sample_times]

    for i in range(len(sample_times)):

        sample_index = sample_indices[i]
        sample_time = sample_times[i]

        sampled_min[sample_time] = {}
        sampled_max[sample_time] = {}

        for name in results_names:
            sampled_min[sample_time][name], sampled_max[sample_time][name] = sim_set_min_max(list(sims.values()),
                                                                                             name,
                                                                                             idx=sample_index)

        if progress_bar is not None:
            progress_bar.value += 1

    result = {}
    for name in results_names:
        result[name] = dict()
        for trial in trials:
            result[name][trial] = list()

    comp = comparators[comparator]

    input_data = []

    for i in range(len(sample_times)):

        sample_index = sample_indices[i]
        sample_time = sample_times[i]

        for name in results_names:
            for trial in trials:
                input_data.append((
                    sims[trial].extract_var_index(name, sample_index),
                    name, 
                    trial, 
                    sampled_min[sample_time][name],
                    sampled_max[sample_time][name],
                    filter
                ))

    pool = get_pool()
    if pool is None:
        pool = mp.Pool(num_workers)
    par_result = pool.starmap(comp, input_data)
    
    for name, trial, res_avg, res_num in par_result:
        result[name][trial].append((res_avg, res_num))

    if progress_bar is not None:
        progress_bar.value += 1

    input_data.clear()

    for name in results_names:
        for trial in trials:
            sum_comps = 0.0
            num_comps = 0
            for avg_el, num_el in result[name][trial]:
                sum_comps += avg_el * num_el
                num_comps += num_el
            result[name][trial] = sum_comps / num_comps

    return result


def corr_traj(_Xt1, _Xt2):
    m, n = _Xt1.shape[1], _Xt2.shape[1]
    _Xtcorr = np.zeros((m, n))

    for ia in range(m):
        for ib in range(n):
            cc = np.corrcoef(_Xt1[:, ia], _Xt2[:, ib])
            _Xtcorr[ia, ib] = cc[0, 1]

    return _Xtcorr


if has_numba:
    corr_traj = numba.njit(corr_traj)


def _analysis_corr_comp(Xt1, Xt2, trial, name):
    Xtcorr = corr_traj(Xt1, Xt2)
    return trial, name, Xtcorr, np.sort(np.max(Xtcorr, axis=1))


def _analysis_corr(Xt, trial, name):
    niter = Xt.shape[1]
    return _analysis_corr_comp(Xt[:, np.arange(0, np.ceil(niter/2), dtype=int)], 
                               Xt[:, np.arange(np.ceil(niter/2), niter, dtype=int)],
                               trial, name)


def analysis_corr_comp(_sims1, _sims2, trials, names):

    num_steps = len(list(_sims1.values())[0].time)
    num_jobs = len(trials) * len(names)
    num_workers = min(mp.cpu_count(), num_jobs)

    jobs = []
    for trial in trials:
        for name in names:
            Xt1 = np.zeros((num_steps, trial), dtype=float)
            Xt2 = np.zeros((num_steps, trial), dtype=float)
            for idx in range(num_steps):
                Xt1[idx, :] = _sims1[trial].extract_var_index(name, idx)
                Xt2[idx, :] = _sims2[trial].extract_var_index(name, idx)
            jobs.append((Xt1, Xt2, trial, name))

    pool = get_pool()
    if pool is None:
        pool = mp.Pool(num_workers)
    par_results = pool.starmap(_analysis_corr_comp, jobs)
    jobs.clear()

    out = {trial: dict() for trial in trials}
    for trial, name, Xtcorr, XtcorrMax in par_results:
        out[trial][name] = Xtcorr, XtcorrMax
    return out


def analysis_corr(_sims, trials, names):

    num_steps = len(list(_sims.values())[0].time)
    num_jobs = len(trials) * len(names)
    num_workers = min(mp.cpu_count(), num_jobs)

    jobs = []
    for trial in trials:
        for name in names:
            Xt = np.zeros((num_steps, trial), dtype=float)
            for idx in range(num_steps):
                Xt[idx, :] = _sims[trial].extract_var_index(name, idx)
            jobs.append((Xt, trial, name))

    pool = get_pool()
    if pool is None:
        pool = mp.Pool(num_workers)
    par_results = pool.starmap(_analysis_corr, jobs)
    jobs.clear()

    out = {trial: dict() for trial in trials}
    for trial, name, Xtcorr, XtcorrMax in par_results:
        out[trial][name] = Xtcorr, XtcorrMax
    return out


def _find_ecfs(_results: Dict[str, np.ndarray], 
               trial: int,
               idx: int,
               num_steps: int,
               num_var_pers: int):
    ks_stat, eval_fin, eval_num = ecf_sample(_results, num_steps, num_var_pers)
    res_ecfs = {n: ecf(_results[n], get_eval_info_times((eval_num[n], eval_fin[n]))) for n in _results.keys()}
    return trial, idx, res_ecfs, ks_stat, eval_fin, eval_num


def find_ecfs(_sims: Dict[int, SimSet], 
              results_names: List[str], 
              trials: List[int], 
              num_steps: int = None,
              num_var_pers: int = None,
              num_workers: int = None,
              quiet: bool = True):
    sample_times = {trial: _sims[trial].results_time for trial in trials}
    result_ecf = {trial: [{} for _ in sample_times[trial]] for trial in trials}
    result_ks_stat = {trial: [{} for _ in sample_times[trial]] for trial in trials}
    eval_info = {t: [{} for _ in sample_times[t]] for t in trials}

    if num_steps is None:
        num_steps = DEF_EVAL_NUM
    if num_var_pers is None:
        num_var_pers = DEF_NUM_VAR_PERS

    input_args = []
    for trial in trials:
        for i, t in enumerate(sample_times[trial]):
            idx = _sims[trial].get_time_index(t)
            input_args.append((
                {n: _sims[trial].results[n].T[idx, :] for n in results_names},
                trial,
                i,
                num_steps,
                num_var_pers
            ))

    if not quiet:
        print(f'Finding {len(input_args)} ECFs')

    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = min(num_workers, len(input_args))

    if not quiet:
        print(f'Using {num_workers} workers')

    pool = get_pool()
    if pool is None:
        pool = mp.Pool(num_workers)
    result_p = pool.starmap(_find_ecfs, input_args)
    
    for trial, idx, res_ecf, ks_stat, eval_fin, eval_num in result_p:
        if not quiet:
            print(f'Got trial: {trial}, {idx}')
        result_ecf[trial][idx] = res_ecf
        result_ks_stat[trial][idx] = ks_stat
        eval_info[trial][idx] = {n: (eval_num[n], eval_fin[n]) for n in eval_num.keys()}

    return result_ecf, result_ks_stat, eval_info


def _generate_ecfs(trial, name, t_num, t_fin, idx, res):
    return trial, name, idx, ecf(res, get_eval_info_times((t_num, t_fin)))


def generate_ecfs(_sims: Dict[int, SimSet], 
                  results_names: List[str], 
                  trials: List[int], 
                  ecf_eval_info: Dict[int, List[Dict[str, ECFEvalInfo]]] = None):
    sample_times = {trial: _sims[trial].results_time for trial in trials}
    if ecf_eval_info is None:
        ecf_eval_info = {t: [{n: (DEF_EVAL_NUM, DEF_EVAL_FIN) for n in results_names} for _ in sample_times[t]]
                         for t in trials}

    result = {trial: [{name: np.ndarray((ecf_eval_info[trial][i][name][0], 2)) for name in results_names} 
                      for i in range(len(sample_times[trial]))]
              for trial in trials}

    input_args = []
    for trial in trials:
        for name in results_names:
            for k, sample_time in enumerate(sample_times[trial]):
                input_args.append((
                    trial, name, ecf_eval_info[trial][k][name][0], ecf_eval_info[trial][k][name][1], k,
                    _sims[trial].extract_var_time(name, sample_time)
                ))

    num_workers = min(mp.cpu_count(), len(input_args))

    pool = get_pool()
    if pool is None:
        pool = mp.Pool(num_workers)
    results_par = pool.starmap(_generate_ecfs, input_args)

    for trial, name, idx, res in results_par:
        result[trial][idx][name] = res

    return result


def _measure_ecf_diff(trial: int, 
                      name: str, 
                      res_time: float,
                      res: np.ndarray, 
                      eval_num: int, 
                      eval_fin: float):
    eval_t = get_eval_info_times((eval_num, eval_fin))
    n = int(res.shape[0] / 2)
    ecf1 = ecf(res[:n], eval_t)
    ecf2 = ecf(res[n:], eval_t)
    return trial, name, res_time, ecf_compare(ecf1[:, 0], ecf1[:, 1], ecf2[:, 0], ecf2[:, 1])


def measure_ecf_diff(_sims: Dict[int, SimSet], 
                     results_names, 
                     trials, 
                     eval_info: Dict[int, List[Dict[str, ECFEvalInfo]]] = None,
                     num_workers: int = None):
    if eval_info is None:
        eval_info = {name: (DEF_EVAL_NUM, DEF_EVAL_FIN) for name in results_names}
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    input_args = []
    for trial in trials:
        sim = _sims[trial]
        for t in sim.time:
            t_idx = sim.get_time_index(t)
            for name in results_names:
                input_args.append((trial, 
                                   name, 
                                   t,
                                   sim.extract_var_index(name, t_idx), 
                                   eval_info[trial][t_idx][name][0], 
                                   eval_info[trial][t_idx][name][1]))

    _num_workers = min(num_workers, len(input_args))

    pool = get_pool()
    if pool is None:
        pool = mp.Pool(_num_workers)
    results_par = pool.starmap(_measure_ecf_diff, input_args)

    result = {trial: {name: list() for name in results_names} for trial in trials}
    for trial, name, ecfc in results_par:
        result[trial][name].append(ecfc)
    for trial in result.keys():
        result[trial] = {k: max(v) for k, v in result[trial].items()}
    return result


def measure_stats(_sims: Dict[int, SimSet]):
    result_mean = {k: dict() for k in _sims.keys()}
    result_stdev = {k: dict() for k in _sims.keys()}

    for trial, sim in _sims.items():
        for name, result in sim.results.items():
            result_mean[trial][name] = np.average(result, 0)
            result_stdev[trial][name] = np.std(result, 0)

    return result_mean, result_stdev


def _measure_ecf_diff_sets(trial: int, 
                           idx: int,
                           name: str, 
                           ecf1: np.ndarray, 
                           ecf2: np.ndarray):
    
    return trial, idx, name, ecf_compare(ecf1[:, 0], ecf1[:, 1], ecf2[:, 0], ecf2[:, 1])


def measure_ecf_diff_sets(_ecf1: Dict[int, List[Dict[str, np.ndarray]]],
                          _ecf2: Dict[int, List[Dict[str, np.ndarray]]],
                          num_workers: int = None):
    if num_workers is None:
        num_workers = mp.cpu_count()

    input_args = []
    for trial in _ecf1.keys():
        ecf1_t = _ecf1[trial]
        ecf2_t = _ecf2[trial]
        for i in range(len(ecf1_t)):
            for name in ecf1_t[i].keys():
                input_args.append((trial, i, name, ecf1_t[i][name], ecf2_t[i][name]))

    _num_workers = min(num_workers, len(input_args))

    pool = get_pool()
    if pool is None:
        pool = mp.Pool(_num_workers)
    results_par = pool.starmap(_measure_ecf_diff_sets, input_args)

    result = {trial: [{name: 2.0 for name in v[i].keys()} for i in range(len(v))] for trial, v in _ecf1.items()}
    for trial, idx, name, ecfc in results_par:
        result[trial][idx][name] = ecfc
    return result


def fit_data(fit_func, data_x, data_y, **kwargs):
    return curve_fit(fit_func, data_x, data_y, **kwargs)


def fit_ecf_diff(ecf_diff, results_names, trials, fit_func, **kwargs):
    data_x = trials
    data_y = {name: [ecf_diff[trial][name] for trial in trials] for name in results_names}
    data_p = {}
    data_p_cov = {}
    for name in results_names:
        data_p[name], data_p_cov[name] = fit_data(fit_func, data_x, data_y[name], **kwargs)
    return data_p, data_p_cov


def _ecf_ks_stat(results: np.ndarray,
                 num_steps: int,
                 num_var_pers: int):
    res_std = np.std(results)
    if res_std == 0.0:
        ks_stat = 0.0
    else:

        eval_pts = np.linspace(2 * num_var_pers * np.pi / res_std, num_steps)
        n = results.shape[0] // 2
        ecf1 = ecf(results[:n], eval_pts)
        ecf2 = ecf(results[n:], eval_pts)
        ks_stat = ecf_compare(ecf1[:, 0], ecf1[:, 1], ecf2[:, 0], ecf2[:, 1])

    return ks_stat


if has_numba:
    _ecf_ks_stat = numba.njit(_ecf_ks_stat)


def _test_sampling_impl(_results: np.ndarray,
                        _indices: np.ndarray,
                        _num_times: int,
                        _num_steps: int,
                        _num_var_pers: int):
    _results_copy = _results[_indices]
    err = np.zeros((_num_times,))
    for idx in range(_num_times):
        err[idx] = _ecf_ks_stat(_results_copy[:, idx].T, _num_steps, _num_var_pers)
    return np.max(err)


if has_numba:
    _test_sampling_impl = numba.njit(_test_sampling_impl)


def _test_sampling(_shm_in_info: Dict[str, str],
                   _shm_out_info: str,
                   _shm_out_idx: int,
                   _shm_out_len: int,
                   _arr_shape0: int,
                   _arr_shape1: int,
                   _num_results: int,
                   _num_steps: int,
                   _num_var_pers: int):
    np.random.seed()
    
    # Get shared data
    shm_in = {k: shared_memory.SharedMemory(name=v) for k, v in _shm_in_info.items()}
    shm_out = shared_memory.SharedMemory(name=_shm_out_info)
    shm_out_arr = np.ndarray((_shm_out_len,), dtype=float, buffer=shm_out.buf)
    out_arr = np.zeros((_num_results,), dtype=float)
    _results = [np.ndarray((_arr_shape0, _arr_shape1), dtype=float, buffer=v.buf) for v in shm_in.values()]
    indices = np.asarray(list(range(_arr_shape0)), dtype=int)

    for i in range(_num_results):
        np.random.shuffle(indices)
        out_arr[i] = max([_test_sampling_impl(res, indices, _arr_shape1, _num_steps, _num_var_pers)
                          for res in _results])

    shm_out_arr[_shm_out_idx:_shm_out_idx+_num_results] = out_arr[:]
    return True


def test_sampling(_results: Dict[str, np.ndarray],
                  incr_sampling=100,
                  err_thresh=1E-4,
                  max_sampling: int = None,
                  num_steps: int = DEF_EVAL_NUM, 
                  num_var_pers: int = DEF_NUM_VAR_PERS,
                  quiet=True):
    var_names = list(_results.keys())
    
    # Allocate shared memory
    shm_to = {k: shared_memory.SharedMemory(create=True, size=v.nbytes) for k, v in _results.items()}
    shm_to_arr = {k: np.ndarray(v.shape, dtype=v.dtype, buffer=shm_to[k].buf) for k, v in _results.items()}
    for k, v in _results.items():
        shm_to_arr[k][:] = v[:]
    shm_to_info = {k: v.name for k, v in shm_to.items()}

    from_arr = np.ndarray((incr_sampling, ), dtype=float)
    shm_from = shared_memory.SharedMemory(create=True, size=from_arr.nbytes)
    shm_from_arr = np.ndarray(from_arr.shape, dtype=from_arr.dtype, buffer=shm_from.buf)

    if not quiet:
        print('Allocating shared memory:', [n for n in shm_to_info.values()])

    # Do stuff
    sample_size, num_times = _results[var_names[0]].shape
    
    ks_stats = []
    
    # Do initial work
    
    if not quiet:
        print('Doing initial work:', incr_sampling)
    
    num_workers = min(incr_sampling, mp.cpu_count())
    
    num_jobs = [0 for _ in range(num_workers)]
    jobs_left = int(incr_sampling)
    while jobs_left > 0:
        for i in range(num_workers):
            if jobs_left > 0:
                num_jobs[i] += 1
                jobs_left -= 1
    num_jobs = [n for n in num_jobs.copy() if n > 0]
    num_workers = len(num_jobs)
    job_indices = [ji - num_jobs[i] for i, ji in enumerate(np.cumsum(num_jobs))]
    
    if sum(num_jobs) != incr_sampling:
        raise RuntimeError(f'Scheduled {sum(num_jobs)} jobs, though {incr_sampling} jobs were requested')

    if not quiet:
        print('\tNumber of workers:', num_workers)
        print('\tJob allocation:', num_jobs)

    input_args = [(shm_to_info, 
                   shm_from.name, 
                   job_indices[i], 
                   incr_sampling, 
                   sample_size, 
                   num_times, 
                   num_jobs[i], 
                   num_steps, 
                   num_var_pers) 
                  for i in range(num_workers)]

    if not quiet:
        print('\tInput arguments:')
        [print(f'\t\t{a}') for a in input_args]

    pool = get_pool()
    if pool is None:
        pool = mp.Pool(num_workers)
    pool.starmap(_test_sampling, input_args)
    from_arr[:] = shm_from_arr[:]
    ks_stats.extend(from_arr.tolist())

    if not quiet:
        print(f'\t\tRetrieved {len(ks_stats)} results')

    # Do iterative work

    if not quiet:
        print('Doing iterative work')
    
    ks_avg_curr = np.average(ks_stats)
    iter_cur = 0
    err_curr = err_thresh + 1.0
    while err_curr >= err_thresh:
        if not quiet:
            print(f'\tIteration {iter_cur+1}')
        
        pool.starmap(_test_sampling, input_args)
        from_arr[:] = shm_from_arr[:]
        ks_stats.extend(from_arr.tolist())

        if not quiet:
            print(f'\t\tRetrieved {len(ks_stats)} results')
        
        ks_avg_next = np.average(ks_stats)
        err_curr = abs(ks_avg_next - ks_avg_curr) / ks_avg_curr

        if not quiet:
            print(f'\t\tCurrent stat : {ks_avg_next}')
            print(f'\t\tCurrent error: {err_curr} ({err_thresh})')

        ks_avg_curr = ks_avg_next
        if ks_avg_curr == 0:
            if not quiet:
                print('\t\tZero average. Terminating.')
            
            break
        
        iter_cur += 1
        if max_sampling is not None and len(ks_stats) >= max_sampling:
            if not quiet:
                print('\t\tMaximum sampling accomplied. Terminating.')
            
            break

    # Free shared memory
    for m in shm_to.values():
        m.close()
        m.unlink()
    shm_from.close()
    shm_from.unlink()

    return ks_stats, iter_cur, err_curr
