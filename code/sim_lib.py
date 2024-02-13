import sys
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

from multiprocessing import shared_memory
import antimony
import csv
import numpy as np
import os
from random import randint
from roadrunner import RoadRunner
import scipy
from typing import Dict, List, Optional, Tuple

# Numba seems to only behave well on Windows
if sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
    has_numba = False
else:
    import numba
    has_numba = True


known_sim_algs = [
    'deterministic', 
    'gillespie_ssa'
]

val_types = {
    float.__name__: float,
    int.__name__: int
}


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
    return lib_pool.map(_seed_pool, [tuple()] * num_workers)


def get_pool():
    return lib_pool


def close_pool():
    global lib_pool
    lib_pool = None


class ParamDist:
    def __init__(self,
                 param_name: str,
                 dist_name: str,
                 dist_params: list) -> None:
        self.param_name = param_name
        self.dist_name = dist_name
        self.dist_params = dist_params

    def to_json(self):
        return dict(param_name=self.param_name, 
                    dist_name=self.dist_name,
                    dist_params=self.dist_params,
                    dist_params_types=[type(v).__name__ for v in self.dist_params])

    @staticmethod
    def from_json(data: dict):
        dist_params = data['dist_params']
        dist_params_types = data['dist_params_types']
        return ParamDist(data['param_name'],
                         data['dist_name'],
                         [val_types[dist_params_types[i]](dist_params[i])
                          for i in range(len(dist_params))])


DEF_SIGFIGS = 15


def round_to_sigfigs(_val: float, _sigfigs: int):
    if _val == 0:
        return _val
    else:
        return np.round(_val, -int(np.multiply(np.sign(_val), np.floor(np.log10(np.abs(_val))))) + _sigfigs - 1)


if has_numba:
    round_to_sigfigs = numba.njit(round_to_sigfigs)


def round_arr_to_sigfigs(_vals: np.ndarray, _sigfigs: int):
    result = np.zeros_like(_vals, dtype=float)
    result_r = result.ravel()
    _vals_r = _vals.ravel()
    for i in range(_vals_r.shape[0]):
        result_r[i] = round_to_sigfigs(_vals_r[i], _sigfigs)
    return result


if has_numba:
    round_arr_to_sigfigs = numba.njit(round_arr_to_sigfigs)


def load_results_copasi(_fp: str, _sample_size: int, _num_steps: int):
    reading_time = True
    sample_times = np.zeros((_num_steps,))
    results = np.zeros((_sample_size, _num_steps))

    trial_idx = 0
    step_idx = 0

    with open(_fp, 'r') as f:
        data_reader = csv.reader(f, delimiter='\t')
        data_reader.__next__()

        for row_t, row_v, _ in data_reader:
            if row_t == 'nan':
                trial_idx += 1
                step_idx = 0
                reading_time = False
                if trial_idx >= results.shape[0]:
                    break
                continue
            else:
                if reading_time:
                    sample_times[step_idx] = float(row_t)
                results[trial_idx, step_idx] = float(row_v)
                step_idx += 1

    if trial_idx != _sample_size:
        raise RuntimeError(f'Got {trial_idx} sample size but expected {_sample_size}')
    return sample_times, results


class Metadata:

    def __init__(self,
                 sample_size: int, 
                 simulator: str, 
                 ks_stat_mean: float,
                 ks_stat_stdev: float,
                 sample_times: np.ndarray,
                 ecf_evals: Dict[str, List[np.ndarray]],
                 ecf_eval_info: Dict[str, List[Tuple[int, float]]],
                 param_dists: List[ParamDist] = None,
                 sig_figs: int = DEF_SIGFIGS) -> None:
        if simulator not in known_sim_algs:
            raise ValueError(f'Unknown simulator algorithm: {simulator}')
        self.sample_size = sample_size
        self.simulator = simulator
        self.ks_stat_mean = ks_stat_mean
        self.ks_stat_stdev = ks_stat_stdev
        self.sample_times = sample_times
        self.ecf_evals = ecf_evals
        self.ecf_eval_info = ecf_eval_info
        self.param_dists = param_dists
        self.sig_figs = sig_figs

    def __str__(self) -> str:
        sl = [f'Sample size: {self.sample_size}',
              f'Simulator: {self.simulator}',
              f'K-S statistic: {self.ks_stat_mean}, {self.ks_stat_stdev}',
              f'No. sample times: {self.sample_times.shape[0]}',
              f'Variables: {list(self.ecf_evals.keys())}',
              f'Significant figures: {self.sig_figs}']
        if self.param_dists is not None:
            sl.append(f'Parameters sampled: {[pd.param_name for pd in self.param_dists]}')
        return '\n'.join(sl)

    def to_json(self):
        data = dict(sample_size=self.sample_size,
                    simulator=self.simulator,
                    ks_stat_mean=self.ks_stat_mean,
                    ks_stat_stdev=self.ks_stat_stdev,
                    sample_times=self.sample_times.tolist(),
                    ecf_evals={k: [vv.tolist() for vv in v] for k, v in self.ecf_evals.items()},
                    ecf_eval_info=self.ecf_eval_info,
                    sig_figs=self.sig_figs)
        if self.param_dists is not None:
            data['param_dists'] = [v.to_json() for v in self.param_dists]
        return data

    @staticmethod
    def from_json(data: dict):
        param_dists = None
        if 'param_dists' in data.keys():
            param_dists = [ParamDist.from_json(d) for d in data['param_dists']]
        return Metadata(int(data['sample_size']),
                        data['simulator'],
                        float(data['ks_stat_mean']),
                        float(data['ks_stat_stdev']),
                        np.array(data['sample_times'], dtype=float),
                        {k: [np.array(vv, dtype=float) for vv in v] for k, v in data['ecf_evals'].items()},
                        {k: [(int(vv[0]), float(vv[1])) for vv in v] for k, v in data['ecf_eval_info'].items()},
                        param_dists,
                        int(data['sig_figs']))


def antimony_to_sbml(_model_string_antimony: str) -> str:
    antimony.clearPreviousLoads()
    antimony.loadAntimonyString(_model_string_antimony)
    module_name = antimony.getMainModuleName()
    return antimony.getSBMLString(module_name)


def make_rr(_model_string_sbml, _stochastic):
    _rr = RoadRunner(_model_string_sbml)

    if _stochastic:
        _rr.integrator = 'gillespie'
        _rr.integrator.seed = randint(0, int(1E9))

    return _rr


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


def mod_rr(_rr: RoadRunner, _mods: dict):
    for k, v in _mods.items():
        _rr[k] = v


def exec_rr(_rr: RoadRunner, 
            _t_fin: float, 
            _num_steps: int, 
            _stochastic: bool, 
            _mods: dict = None, 
            _dists: dict = None):
    while True:
        _rr.resetAll()
        # if _stochastic:
        #     _rr.integrator.integrator = 'gillespie'
        #     _rr.integrator.seed = randint(0, int(1E6))
        if _mods is not None:
            mod_rr(_rr, _mods)
        if _dists is not None:
            param_dists = generate_dists(_dists)
            apply_dists(_rr, param_dists)
        try:
            return _rr.simulate(0.0, _t_fin, _num_steps)
        except Exception as e:
            print(e)
            pass


def exec_rr_batch(_num_jobs: int, 
                  _rr: RoadRunner, 
                  _t_fin: float, 
                  _num_steps: int, 
                  _stochastic: bool, 
                  mods: dict = None, 
                  dists: dict = None, 
                  out=None):
    
    sbml_string = _rr.getSBML()
    args = [(make_rr(sbml_string, _stochastic), _t_fin, _num_steps, _stochastic, mods, dists) for _ in range(_num_jobs)]
    pool = get_pool()
    if pool is None:
        num_workers = min(_num_jobs, mp.cpu_count())
        if out is not None:
            out.append_stdout(f'Starting pool: {num_workers}\n')
        pool = mp.Pool(num_workers)

    result = pool.starmap(exec_rr, args)
    if out is not None:
        out.append_stdout('Terminating pool\n')

    return result


DEF_EVAL_NUM = 100
DEF_NUM_VAR_PERS = 5


def _ecf_eval_pts(num_steps: int, incr: float) -> np.ndarray:
    return np.arange(0.0, (num_steps+1) * incr, incr)


if has_numba:
    _ecf_eval_pts = numba.njit(_ecf_eval_pts)


def ecf_compare(_ecf_1: np.ndarray, _ecf_2: np.ndarray):
    return np.max(np.sqrt(np.square(_ecf_1[:, 0] - _ecf_2[:, 0]) + np.square(_ecf_1[:, 1] - _ecf_2[:, 1])))


if has_numba:
    ecf_compare = numba.njit(ecf_compare)


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


def _ecf_ks_stat(results: np.ndarray,
                 num_steps: int,
                 num_var_pers: int):
    if np.std(results) == 0.0:
        incr_max = 1 / num_steps
        ks_stat = 0.0
    else:

        incr_max = 2 * num_var_pers * np.pi / np.std(results) / num_steps

        eval_pts = _ecf_eval_pts(num_steps, incr_max)
        n = int(results.shape[0] / 2)
        ecf1 = ecf(results[:n], eval_pts)
        ecf2 = ecf(results[n:], eval_pts)
        ks_stat = ecf_compare(ecf1, ecf2)

    return incr_max, ks_stat


if has_numba:
    _ecf_ks_stat = numba.njit(_ecf_ks_stat)


def get_eval_info_times(_eval_num: int, _eval_fin: float, stagger=True):
    if not stagger:
        return np.arange(0.0, _eval_fin * (1 + 1 / _eval_num), _eval_fin / _eval_num)
    
    idx = np.asarray(list(range(_eval_num+1)))
    h = _eval_fin / _eval_num
    mask = np.mod(idx, 2) != 0
    result = idx * h
    result[mask] += ((idx[mask] + 1) * 2 / (_eval_num + 2) - 1) * h
    return result


if has_numba:
    get_eval_info_times = numba.njit(get_eval_info_times)


def ecf_sample(_results: Dict[str, np.ndarray], 
               num_steps=DEF_EVAL_NUM,
               num_var_pers: int = DEF_NUM_VAR_PERS):
    res_ks_stat = {}
    res_eval_fin = {}
    for k, res in _results.items():
        incr_max, res_ks_stat[k] = _ecf_ks_stat(res, num_steps, num_var_pers)
        res_eval_fin[k] = incr_max * num_steps

    res_num_steps = {n: num_steps for n in _results.keys()}

    return res_ks_stat, res_eval_fin, res_num_steps


def _find_ecfs(_results: Dict[str, np.ndarray], 
               idx: int,
               num_steps: int,
               num_var_pers: int):
    ks_stat, eval_fin, eval_num = ecf_sample(_results, num_steps, num_var_pers)
    res_ecfs = {n: ecf(_results[n], get_eval_info_times(eval_num[n], eval_fin[n])) for n in _results.keys()}
    return idx, res_ecfs, ks_stat, eval_fin, eval_num


def find_ecfs(_results: Dict[str, np.ndarray], 
              num_steps: int = None,
              num_var_pers: int = None,
              num_workers: int = None,
              quiet: bool = True):
    result_ecf = {name: [None for _ in range(_results[name].shape[1])] for name in _results.keys()}
    result_ks_stat = {name: [None for _ in range(_results[name].shape[1])] for name in _results.keys()}
    eval_info = {name: [None for _ in range(_results[name].shape[1])] for name in _results.keys()}

    if num_steps is None:
        num_steps = DEF_EVAL_NUM
    if num_var_pers is None:
        num_var_pers = DEF_NUM_VAR_PERS

    input_args = []
    for idx in range(_results[list(_results.keys())[0]].shape[1]):
        input_args.append((
            {name: _results[name][:, idx].T for name in _results.keys()},
            idx,
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

    with mp.Pool(num_workers) as p:
        result_p = p.starmap(_find_ecfs, input_args)
    
    for idx, res_ecf, ks_stat, eval_fin, eval_num in result_p:
        if not quiet:
            print(f'Got ECF: {idx}')
        for name in res_ecf.keys():
            result_ecf[name][idx] = res_ecf[name]
            result_ks_stat[name][idx] = ks_stat[name]
            eval_info[name][idx] = (eval_num[name], eval_fin[name])

    return result_ecf, result_ks_stat, eval_info


def recommend(trials, outcomes, target_outcome):
    fit_func = lambda n, a, b: a * n ** b
    fit_a, fit_b = scipy.optimize.curve_fit(fit_func, trials, outcomes)[0]
    return (target_outcome / fit_a) ** (1. / fit_b)


def _test_sampling_impl(_results: np.ndarray,
                        _num_times: int, 
                        _num_steps: int, 
                        _num_var_pers: int):
    ks_stat_iter = 0.0
    for idx in range(_num_times):
        ks_stat_iter = max(ks_stat_iter, _ecf_ks_stat(_results[:, idx].T, _num_steps, _num_var_pers)[1])
    return ks_stat_iter


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
                   _num_var_pers: int) -> bool:
    # Ensure worker has unique seed
    np.random.seed()
    # Get shared data
    shm_in = {k: shared_memory.SharedMemory(name=v) for k, v in _shm_in_info.items()}
    shm_out = shared_memory.SharedMemory(name=_shm_out_info)

    _results = {k: np.ndarray((_arr_shape0, _arr_shape1), dtype=float, buffer=v.buf) for k, v in shm_in.items()}
    shm_out_arr = np.ndarray((_shm_out_len,), dtype=float, buffer=shm_out.buf)
    out_arr = np.zeros((_num_results,), dtype=float)
    results_copy = [np.array(v) for v in _results.values()]
    indices = np.asarray(list(range(_arr_shape0)), dtype=int)

    for i in range(_num_results):
        np.random.shuffle(indices)
        results_copy = [res[indices] for res in results_copy]
        max_val = 0.0
        for res in results_copy:
            max_val = max(max_val, _test_sampling_impl(res, _arr_shape1, _num_steps, _num_var_pers))
        out_arr[i] = max_val

    shm_out_arr[_shm_out_idx:_shm_out_idx+_num_results] = out_arr[:]
    return True


def test_sampling(_results: Dict[str, np.ndarray],
                  incr_sampling=100,
                  err_thresh=1E-4,
                  max_sampling: int = None,
                  num_steps: int = DEF_EVAL_NUM, 
                  num_var_pers: int = DEF_NUM_VAR_PERS,
                  out=None,
                  num_workers: int = None):
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

    if out is not None:
        out.append_stdout(f'Allocating shared memory: {[n for n in shm_to_info.values()]}\n')

    # Do stuff
    sample_size, num_times = _results[var_names[0]].shape
    
    ks_stats = []
    
    # Do initial work
    
    if out is not None:
        out.append_stdout(f'Doing initial work: {incr_sampling}\n')
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = min(incr_sampling, num_workers)
    
    num_jobs = [0 for _ in range(num_workers)]
    jobs_left = int(incr_sampling)
    while jobs_left > 0:
        for i in range(num_workers):
            if jobs_left > 0:
                num_jobs[i] += 1
                jobs_left -= 1
    num_jobs = [n for n in num_jobs if n > 0]
    num_workers = len(num_jobs)
    job_indices = [ji - num_jobs[i] for i, ji in enumerate(np.cumsum(num_jobs))]
    
    if sum(num_jobs) != incr_sampling:
        raise RuntimeError(f'Scheduled {sum(num_jobs)} jobs, though {incr_sampling} jobs were requested')

    if out is not None:
        out.append_stdout(f'Number of workers: {num_workers}; Job allocation: {num_jobs}\n')

    pool = get_pool()
    if pool is None:
        if out is not None:
            out.append_stdout('Launching pool\n')
        pool = mp.Pool(num_workers)

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

    if out is not None:
        out.append_stdout('Starting pool\n')
    pool.starmap(_test_sampling, input_args)
    if out is not None:
        out.append_stdout('Terminating pool\n')
    
    from_arr[:] = shm_from_arr[:]
    ks_stats.extend(from_arr.tolist())

    if out is not None:
        out.append_stdout(f'Retrieved {len(ks_stats)} results\n')

    # Do iterative work

    if out is not None:
        out.append_stdout('Doing iterative work\n')
    
    ks_avg_curr = np.average(ks_stats)
    iter_cur = 0
    err_curr = err_thresh + 1.0
    while err_curr >= err_thresh:
        if out is not None:
            out.append_stdout(f'Iteration {iter_cur+1}\n')
        
        if out is not None:
            out.append_stdout('Starting pool\n')
        pool.starmap(_test_sampling, input_args)
        if out is not None:
            out.append_stdout('Terminating pool\n')
        
        from_arr[:] = shm_from_arr[:]
        ks_stats.extend(from_arr.tolist())

        if out is not None:
            out.append_stdout(f'Retrieved {len(ks_stats)} results\n')
        
        ks_avg_next = np.average(ks_stats)
        err_curr = abs(ks_avg_next - ks_avg_curr) / ks_avg_curr

        if out is not None:
            out.append_stdout(f'Current stat : {ks_avg_next}\n')
            out.append_stdout(f'Current error: {err_curr} ({err_thresh})\n')

        ks_avg_curr = ks_avg_next
        if ks_avg_curr == 0:
            if out is not None:
                out.append_stdout('Zero average. Terminating.\n')
            
            break
        
        iter_cur += 1
        if max_sampling is not None and len(ks_stats) >= max_sampling:
            if out is not None:
                out.append_stdout('Maximum sampling accomplied. Terminating.\n')
            
            break

    # Free shared memory
    for m in shm_to.values():
        m.close()
        m.unlink()
    shm_from.close()
    shm_from.unlink()

    # return result
    return ks_stats, iter_cur, err_curr


def _test_sampling_impl_no_shared(_results: np.ndarray,
                        _num_times: int, 
                        _num_steps: int, 
                        _num_var_pers: int):
    ks_stat_iter = 0.0
    for idx in range(_num_times):
        ks_stat_iter = max(ks_stat_iter, _ecf_ks_stat(_results[:, idx].T, _num_steps, _num_var_pers)[1])
    return ks_stat_iter


if has_numba:
    _test_sampling_impl_no_shared = numba.njit(_test_sampling_impl_no_shared)


def _test_sampling_no_shared(_results: Dict[str, np.ndarray],
                             _arr_shape0: int,
                             _arr_shape1: int,
                             _num_results: int,
                             _num_steps: int, 
                             _num_var_pers: int):
    # Ensure worker has unique seed
    np.random.seed()
    
    indices = np.asarray(list(range(_arr_shape0)), dtype=int)
    result = []

    for _ in range(_num_results):
        np.random.shuffle(indices)
        result.append(max([_test_sampling_impl_no_shared(res[indices, :], _arr_shape1, _num_steps, _num_var_pers) for res in _results.values()]))

    return result


def test_sampling_no_shared(_results: Dict[str, np.ndarray],
                            incr_sampling=100,
                            err_thresh=1E-4,
                            max_sampling: int = None,
                            num_steps: int = DEF_EVAL_NUM, 
                            num_var_pers: int = DEF_NUM_VAR_PERS,
                            out=None,
                            num_workers: int = None):
    var_names = list(_results.keys())
    
    # Do stuff
    sample_size, num_times = _results[var_names[0]].shape
    
    ks_stats = []
    
    # Do initial work
    
    if out is not None:
        out.append_stdout(f'Doing initial work: {incr_sampling}\n')
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = min(incr_sampling, num_workers)
    
    num_jobs = [0 for _ in range(num_workers)]
    jobs_left = int(incr_sampling)
    while jobs_left > 0:
        for i in range(num_workers):
            if jobs_left > 0:
                num_jobs[i] += 1
                jobs_left -= 1
    num_jobs = [n for n in num_jobs if n > 0]
    num_workers = len(num_jobs)
    
    if sum(num_jobs) != incr_sampling:
        raise RuntimeError(f'Scheduled {sum(num_jobs)} jobs, though {incr_sampling} jobs were requested')

    if out is not None:
        out.append_stdout(f'Number of workers: {num_workers}; Job allocation: {num_jobs}\n')

    pool = get_pool()
    if pool is None:
        if out is not None:
            out.append_stdout('Launching pool\n')
        pool = mp.Pool(num_workers)

    input_args = [(_results,
                   sample_size, 
                   num_times, 
                   num_jobs[i], 
                   num_steps, 
                   num_var_pers) 
                  for i in range(num_workers)]

    if out is not None:
        out.append_stdout('Starting pool\n')
    [ks_stats.extend(res) for res in pool.starmap(_test_sampling_no_shared, input_args)]
    if out is not None:
        out.append_stdout('Terminating pool\n')

    if out is not None:
        out.append_stdout(f'Retrieved {len(ks_stats)} results\n')

    # Do iterative work

    if out is not None:
        out.append_stdout('Doing iterative work\n')
    
    ks_avg_curr = np.average(ks_stats)
    iter_cur = 0
    if ks_avg_curr == 0:
        if out is not None:
            out.append_stdout('Zero average. Terminating.\n')
        err_curr = 0.0
    else:
        err_curr = err_thresh + 1.0
    while err_curr >= err_thresh:
        if out is not None:
            out.append_stdout(f'Iteration {iter_cur+1}\n')
        
        if out is not None:
            out.append_stdout('Starting pool\n')
        [ks_stats.extend(res) for res in pool.starmap(_test_sampling_no_shared, input_args)]
        if out is not None:
            out.append_stdout('Terminating pool\n')

        if out is not None:
            out.append_stdout(f'Retrieved {len(ks_stats)} results\n')
        
        ks_avg_next = np.average(ks_stats)
        err_curr = abs(ks_avg_next - ks_avg_curr) / ks_avg_curr

        if out is not None:
            out.append_stdout(f'Current stat : {ks_avg_next}\n')
            out.append_stdout(f'Current error: {err_curr} ({err_thresh})\n')

        ks_avg_curr = ks_avg_next
        if ks_avg_curr == 0:
            if out is not None:
                out.append_stdout('Zero average. Terminating.\n')
            
            break
        
        iter_cur += 1
        if max_sampling is not None and len(ks_stats) >= max_sampling:
            if out is not None:
                out.append_stdout('Maximum sampling accomplied. Terminating.\n')
            
            break

    # return result
    return ks_stats, iter_cur, err_curr