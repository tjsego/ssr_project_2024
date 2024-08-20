import json
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from scipy import stats
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import sim_lib

if sim_lib.has_numba:
    import numba

# Hack in algorithm
alg_name = 'sde'
sim_lib.known_sim_algs.append(alg_name)


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


class SDEModel:

    name = ''
    variable_names: List[str] = []
    parameter_defaults: Dict[str, float] = {}

    def __init__(self, *args, **kwargs):
        pass

    def to_json(self):
        return dict(name=self.name)

    @classmethod
    def from_json(cls, _data: dict):
        load_models()
        args = _data.get('args', [])
        kwargs = _data.get('kwargs', {})
        return __models__[_data['name']](*args, **kwargs)

    def __reduce__(self):
        return SDEModel.from_json, (self.to_json(),)

    def step(self,
             current_time: float,
             current_vals: np.ndarray,
             dt: float,
             parameters: Dict[str, float]) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def register_model(cls):
        global __models__
        __models__[cls.name] = cls

    @staticmethod
    def get(name: str, *args, **kwargs):
        load_models()
        return __models__[name](*args, **kwargs)


__models__: Dict[str, Type[SDEModel]] = {}


def load_models():
    import sde_models
    for model in sde_models.__known_models__:
        model.register_model()


class SDEResult:

    def __init__(self, var_names: List[str]):

        self.var_names = var_names
        self.time = np.ndarray(())
        self.data = np.ndarray(())

    def extend(self, other):
        other: SDEResult

        if self.var_names != other.var_names:
            raise ValueError

        self.time = np.hstack((self.time, other.time))
        self.data = np.vstack((self.data, other.data))

    def copy(self):
        result = SDEResult(self.var_names.copy())
        result.time = np.array(self.time)
        result.data = np.array(self.data)
        return result

    def extract(self, *names):
        result = SDEResult(list(names))
        result.time = np.array(self.time)
        result.data = np.zeros((len(names), self.data.shape[1]))
        ind_result = result.indices
        ind_self = self.indices
        for n in names:
            result.data[ind_result[n], :] = self.data[ind_self[n], :]
        return result

    @property
    def indices(self):
        return {name: i for i, name in enumerate(self.var_names)}

    def index(self, name: str):
        return self.indices[name]

    def initialize(self,
                   num_steps: int,
                   initial_values: Dict[str, float] = None):

        self.time = np.zeros((num_steps + 1,), dtype=float)
        self.data = np.zeros((len(self.var_names), num_steps + 1), dtype=float)
        if initial_values is not None:
            indices = self.indices
            for k, v in initial_values.items():
                self.data[indices[k], 0] = v

    def values_by_name(self, name: str):
        return self.data[self.indices[name], :]

    def data_by_index(self, idx: int):
        return self.time[idx], self.data[:, idx]

    def to_json(self):
        return dict(
            var_names=self.var_names,
            time=self.time.tolist(),
            data=self.data.tolist()
        )

    @classmethod
    def from_json(cls, _data: dict):
        result = cls(_data['var_names'])
        result.time = np.asarray(_data['time'], dtype=float)
        result.data = np.asarray(_data['data'], dtype=float)
        return result

    def __reduce__(self):
        return SDEResult.from_json, (self.to_json(),)

    def plot(self,
             fig_kwargs: dict = None,
             fig_axs=None,
             plot_kwargs: dict = None,
             plot_all_kwargs: dict = None):
        if self.time.shape[0] == 0:
            raise RuntimeError

        if fig_axs is None:
            fig_axs = plt.subplots(1, len(self.var_names), **fig_kwargs)

        plot_kwargs_actual = {name: {} for name in self.var_names}
        if plot_all_kwargs is not None:
            [plot_kwargs_actual[name].update(plot_all_kwargs) for name in self.var_names]
        if plot_kwargs is not None:
            [plot_kwargs_actual[name].update(v) for name, v in plot_kwargs.items()]

        for i, name in enumerate(self.var_names):
            ax = fig_axs[1][i]
            ax.plot(self.time, self.data[i, :], **plot_kwargs_actual[name])
            ax.set_xlabel('Time')
            ax.set_title(name)

        return fig_axs


def _compare_samples(sample1: np.ndarray, sample2: np.ndarray):
    err = 0.0
    for j in range(sample1.shape[1]):
        for k in range(sample1.shape[2]):
            std_res = np.std(sample1[:, j, k])
            eval_t_fin = 2.0 * np.pi * sim_lib.DEF_NUM_VAR_PERS / std_res if std_res != 0.0 else 1.0
            eval_t = sim_lib.get_eval_info_times(sim_lib.DEF_EVAL_NUM, eval_t_fin)
            ecf1 = sim_lib.ecf(sample1[:, j, k], eval_t)
            ecf2 = sim_lib.ecf(sample2[:, j, k], eval_t)
            err = max(err, sim_lib.ecf_compare(ecf1, ecf2))
    return err


def _compare_samples_nb(sample1: np.ndarray, sample2: np.ndarray):
    err = 0.0
    err_j = np.zeros((sample1.shape[2],))
    for j in range(sample1.shape[1]):
        for k in numba.prange(sample1.shape[2]):
            std_res = np.std(sample1[:, j, k])
            eval_t_fin = 2.0 * np.pi * sim_lib.DEF_NUM_VAR_PERS / std_res if std_res != 0.0 else 1.0
            eval_t = sim_lib.get_eval_info_times(sim_lib.DEF_EVAL_NUM, eval_t_fin)
            ecf1 = sim_lib.ecf(sample1[:, j, k], eval_t)
            ecf2 = sim_lib.ecf(sample2[:, j, k], eval_t)
            err_j[k] = sim_lib.ecf_compare(ecf1, ecf2)
        err = max(err, np.max(err_j))
    return err


if sim_lib.has_numba:
    _compare_samples = numba.njit(_compare_samples_nb, parallel=True)


class SDEResultSample:

    def __init__(self, var_names: List[str], sample_size: int, num_times: int):

        self.var_names = var_names
        self.time = np.zeros((num_times,), dtype=float)
        self.data = np.zeros((sample_size, len(var_names), num_times), dtype=float)

    @staticmethod
    def compare_samples(sample1, sample2):
        sample1: SDEResultSample
        sample2: SDEResultSample

        if sample1.data.shape != sample2.data.shape:
            raise RuntimeError('Samples are not comparable')

        return _compare_samples(sample1.data, sample2.data)

    def copy(self):
        result = SDEResultSample(self.var_names.copy(), self.data.shape[0], self.time.shape[0])
        result.time = np.array(self.time)
        result.data = np.array(self.data)
        return result

    def extract(self, *names):
        result = SDEResultSample(list(names), len(self), self.time.shape[0])
        result.time = np.array(self.time)
        ind_result = result.indices
        ind_self = self.indices
        for n in names:
            result.data[:, ind_result[n], :] = self.data[:, ind_self[n], :]
        return result

    @property
    def indices(self):
        return {name: i for i, name in enumerate(self.var_names)}

    def index(self, name: str):
        return self.indices[name]

    def to_json(self):
        return dict(
            var_names=self.var_names,
            time=self.time.tolist(),
            data=self.data.tolist()
        )

    @classmethod
    def from_json(cls, _data):
        time = np.asarray(_data['time'], dtype=float)
        data = np.asarray(_data['data'], dtype=float)

        result = cls(_data['var_names'], data.shape[0], data.shape[2])
        result.time = time
        result.data = data
        return result

    def __reduce__(self):
        return SDEResultSample.from_json, (self.to_json(),)

    def __getitem__(self, item: int):
        result = SDEResult(self.var_names)
        result.time = self.time
        result.data = self.data[item, :, :]
        return result

    def __setitem__(self, key: int, value: SDEResult):
        self.data[key, :, :] = value.data

    def __len__(self):
        return self.data.shape[0]

    def generate_metadata(self, sig_figs=16) -> sim_lib.Metadata:
        indices = self.indices

        results = {name: sim_lib.round_arr_to_sigfigs(self.data[:, indices[name], :], sig_figs)
                   for name in self.var_names}
        ks_stats_samp_hist = sim_lib.test_sampling(results, err_thresh=1E-3)[0]

        sample_size = int(self.data.shape[0] / 2)
        results_export = {name: v[:sample_size, :] for name, v in results.items()}
        ecf_evals, _, ecf_eval_info = sim_lib.find_ecfs(results_export)

        return sim_lib.Metadata(sample_size=sample_size,
                                simulator=alg_name,
                                ks_stat_mean=np.mean(ks_stats_samp_hist),
                                ks_stat_stdev=np.std(ks_stats_samp_hist, ddof=1),
                                sample_times=self.time,
                                ecf_evals=ecf_evals,
                                ecf_eval_info=ecf_eval_info,
                                sig_figs=sig_figs)

    @property
    def mean(self):
        return np.average(self.data, axis=0)

    @property
    def std(self):
        return np.std(self.data, axis=0)

    def ci_inc(self, confidence=0.95):
        result = np.zeros((self.data.shape[1], self.data.shape[2]), dtype=float)
        moe_cf = stats.t.ppf((1 + confidence) / 2., self.data.shape[0] - 1)
        for i in range(self.data.shape[1]):
            for j in range(self.data.shape[2]):
                result[i, j] = stats.sem(self.data[:, i, j]) * moe_cf
        return result

    def plot(self,
             fig_kwargs: dict = None,
             fig_axs=None,
             plot_kwargs: dict = None,
             plot_all_kwargs: dict = None):
        if self.time.shape[0] == 0:
            raise RuntimeError

        if fig_axs is None:
            if fig_kwargs is None:
                fig_kwargs = {}
            fig_axs = plt.subplots(1, len(self.var_names), **fig_kwargs)

        plot_kwargs_actual = {name: {} for name in self.var_names}
        if plot_all_kwargs is not None:
            [plot_kwargs_actual[name].update(plot_all_kwargs) for name in self.var_names]
        if plot_kwargs is not None:
            [plot_kwargs_actual[name].update(v) for name, v in plot_kwargs.items()]

        for i in range(len(self)):
            self[i].plot(fig_axs=fig_axs, plot_kwargs=plot_kwargs, plot_all_kwargs=plot_all_kwargs)
        return fig_axs

    def plot_mean(self,
                  n_std: int = 0,
                  fig_kwargs: dict = None,
                  fig_axs=None,
                  plot_kwargs: dict = None,
                  plot_all_kwargs: dict = None,
                  plot_std_kwargs: dict = None):
        if self.time.shape[0] == 0:
            raise RuntimeError

        if fig_axs is None:
            if fig_kwargs is None:
                fig_kwargs = {}
            fig_axs = plt.subplots(1, len(self.var_names), **fig_kwargs)

        plot_kwargs_actual = {name: {} for name in self.var_names}
        if plot_all_kwargs is not None:
            [plot_kwargs_actual[name].update(plot_all_kwargs) for name in self.var_names]
        if plot_kwargs is not None:
            [plot_kwargs_actual[name].update(v) for name, v in plot_kwargs.items()]

        plot_std_kwargs_actual = {k: {kk: vv for kk, vv in v.items()} for k, v in plot_kwargs_actual.items()}
        if plot_std_kwargs is None:
            plot_std_kwargs = {}
        for k, v in plot_std_kwargs_actual.items():
            v.update(plot_std_kwargs)
            if 'linestyle' not in v:
                v['linestyle'] = '--'
            if 'label' in v:
                v.pop('label')

        mean = self.mean
        std = (self.std * n_std) if n_std > 0 else None
        for i, name in enumerate(self.var_names):
            ax = fig_axs[1][i]
            if n_std > 0:
                ax.plot(self.time, mean[i, :] + std[i, :], **plot_std_kwargs_actual[name])
                ax.plot(self.time, mean[i, :] - std[i, :], **plot_std_kwargs_actual[name])
            ax.plot(self.time, mean[i, :], **plot_kwargs_actual[name])
            ax.set_xlabel('Time')
            ax.set_title(name)

        return fig_axs

    def plot_ci(self,
                confidence=0.95,
                fig_kwargs: dict = None,
                fig_axs=None,
                plot_kwargs: dict = None,
                plot_all_kwargs: dict = None):
        if self.time.shape[0] == 0:
            raise RuntimeError

        if fig_axs is None:
            if fig_kwargs is None:
                fig_kwargs = {}
            fig_axs = plt.subplots(1, len(self.var_names), **fig_kwargs)
            if len(self.var_names) == 1:
                fig_axs = (fig_axs[0], [fig_axs[1]])

        plot_kwargs_actual = {name: {} for name in self.var_names}
        if plot_all_kwargs is not None:
            [plot_kwargs_actual[name].update(plot_all_kwargs) for name in self.var_names]
        if plot_kwargs is not None:
            [plot_kwargs_actual[name].update(v) for name, v in plot_kwargs.items()]

        mean = self.mean
        ci_inc = self.ci_inc(confidence)
        ci_p = mean + ci_inc
        ci_n = mean - ci_inc
        for i, name in enumerate(self.var_names):
            ax = fig_axs[1][i]
            ax.fill_between(self.time, ci_n[i, :], ci_p[i, :], **plot_kwargs_actual[name])
            ax.set_xlabel('Time')
            ax.set_title(name)

        return fig_axs


class SDESimulation:

    def __init__(self,
                 model: SDEModel,
                 num_steps: int,
                 dt=1.0,
                 initial_values: Dict[str, float] = None,
                 parameters: Dict[str, float] = None):

        self.model = model
        self.num_steps = num_steps
        self.dt = dt
        self.initial_values = initial_values
        self.parameters = {k: v for k, v in model.parameter_defaults.items()}
        if parameters is not None:
            self.parameters.update(parameters)

    def to_json(self):
        result = dict(
            model=self.model.to_json(),
            num_steps=self.num_steps,
            dt=self.dt
        )
        if self.initial_values is not None:
            result['initial_values'] = self.initial_values
        if self.parameters is not None:
            result['parameters'] = self.parameters
        return result

    @classmethod
    def from_json(cls, _data: dict):
        model = SDEModel.from_json(_data['model'])
        ivs = {k: float(v) for k, v in _data['initial_values'].items()} if 'initial_values' in _data else None
        ps = {k: float(v) for k, v in _data['parameters'].items()} if 'parameters' in _data else None
        return cls(model, int(_data['num_steps']), float(_data['dt']), initial_values=ivs, parameters=ps)

    def run(self):
        result = SDEResult(self.model.variable_names)
        result.initialize(self.num_steps, self.initial_values)
        for i in range(self.num_steps):
            result.time[i + 1] = (i + 1) * self.dt
            result.data[:, i + 1] = self.model.step(result.time[i], result.data[:, i], self.dt, self.parameters)
        return result

    @staticmethod
    def execute_sim(model: SDEModel,
                    num_steps: int,
                    dt=1.0,
                    initial_values: Dict[str, float] = None,
                    parameters: Dict[str, float] = None):
        return SDESimulation(model, num_steps, dt, initial_values, parameters).run()


def execute_sample(sample_size: int,
                   model: SDEModel,
                   num_steps: int,
                   dt=1.0,
                   initial_values: Dict[str, float] = None,
                   parameters: Dict[str, float] = None,
                   pool: mp.Pool = None):
    if pool is None:
        pool = get_pool()
    if pool is None:
        start_pool()
        pool = get_pool()

    sample = SDEResultSample(model.variable_names, sample_size, num_steps + 1)
    input_args = [(model, num_steps, dt, initial_values, parameters)] * sample_size
    for i, result in enumerate(pool.starmap(SDESimulation.execute_sim, input_args)):
        if i == 0:
            sample.time = result.time
        sample[i] = result
    return sample


def evaluate_precision_ics(reference: SDEResultSample,
                           model: SDEModel,
                           name: str,
                           value_ratios: List[float],
                           parameters: Dict[str, float] = None,
                           pool: mp.Pool = None):
    result = [0.0] * len(value_ratios)
    sample_size = len(reference)

    reference_0 = reference[0]
    dt = reference_0.time[1] - reference_0.time[0]
    num_steps = reference_0.time.shape[0] - 1
    initial_values = {name: reference_0.data[reference_0.index(name), 0] for name in reference.var_names}
    var_values = [initial_values[name] * v for v in value_ratios]
    for i, val in enumerate(var_values):
        initial_values[name] = val
        sample = execute_sample(sample_size, model, num_steps, dt,
                                initial_values=initial_values, parameters=parameters, pool=pool)
        result[i] = SDEResultSample.compare_samples(reference, sample)

    return result


def evaluate_precision_params(reference: SDEResultSample,
                              model: SDEModel,
                              name: str,
                              value_ratios: List[float],
                              parameters: Dict[str, float] = None,
                              pool: mp.Pool = None):
    result = [0.0] * len(value_ratios)
    sample_size = len(reference)

    reference_0 = reference[0]
    dt = reference_0.time[1] - reference_0.time[0]
    num_steps = reference_0.time.shape[0] - 1
    initial_values = {name: reference_0.data[reference_0.index(name), 0] for name in reference.var_names}

    parameters_actual = {k: v for k, v in model.parameter_defaults.items()}
    if parameters is not None:
        parameters_actual.update(parameters)

    var_values = [parameters_actual[name] * v for v in value_ratios]
    for i, val in enumerate(var_values):
        parameters_actual[name] = val
        sample = execute_sample(sample_size, model, num_steps, dt,
                                initial_values=initial_values, parameters=parameters_actual, pool=pool)
        result[i] = SDEResultSample.compare_samples(reference, sample)

    return result


def test_precision_params(reference: SDEResultSample,
                          model: SDEModel,
                          name: str,
                          value_ratios: List[float],
                          repro_sampling_stats: Tuple[float, float],
                          parameters: Dict[str, float] = None,
                          pool: mp.Pool = None):
    err_compare = evaluate_precision_params(reference, model, name, value_ratios, parameters, pool)

    err_avg, err_std = repro_sampling_stats
    sample_size = reference.data.shape[0]
    q2 = (sample_size + 1) / sample_size * err_std * err_std

    comparison_pvals = []
    for err in err_compare:
        if err < err_avg:
            pval = 1.0
        else:
            lam2 = ((err - err_avg) * (err - err_avg)) / q2
            pval = np.floor((sample_size + 1) / sample_size * ((sample_size - 1) / lam2 + 1)) / (sample_size + 1)
            if pval > 1:
                pval = 1.0
        comparison_pvals.append(pval)

    return err_compare, comparison_pvals


def plot_precision(repro_sampling_stats: Dict[int, Tuple[float, float]],
                   input_values: List[float],
                   evaluated_errors: Dict[int, List[float]],
                   fig_axs=None,
                   scatter_plot_kwargs: Dict[int, Dict[str, Any]] = None,
                   fill_plot_kwargs: Dict[int, Dict[str, Any]] = None,
                   scatter_plot_all_kwargs: Dict[str, Any] = None,
                   fill_plot_all_kwargs: Dict[str, Any] = None,
                   **fig_kwargs):
    sizes = list(repro_sampling_stats.keys())
    sizes.sort()
    num_sizes = len(sizes)

    if fig_axs is None:
        fig, axs = plt.subplots(num_sizes, 1, sharex=True, sharey=True, **fig_kwargs)
    else:
        fig, axs = fig_axs
    if num_sizes == 1:
        axs = [axs]

    if scatter_plot_kwargs is None:
        scatter_plot_kwargs = {}
    if fill_plot_kwargs is None:
        fill_plot_kwargs = {}
    if scatter_plot_all_kwargs is None:
        scatter_plot_all_kwargs = {}
    if fill_plot_all_kwargs is None:
        fill_plot_all_kwargs = {}
    _scatter_plot_kwargs = {sz: dict() for sz in sizes}
    _fill_plot_kwargs = {sz: dict(color='lightgray', alpha=0.5) for sz in sizes}
    [v.update(scatter_plot_all_kwargs) for v in _scatter_plot_kwargs.values()]
    [v.update(fill_plot_all_kwargs) for v in _fill_plot_kwargs.values()]
    [_scatter_plot_kwargs[k].update(v) for k, v in scatter_plot_kwargs.items()]
    [_fill_plot_kwargs[k].update(v) for k, v in fill_plot_kwargs.items()]

    for i, sz in enumerate(sizes):
        avg, err = repro_sampling_stats[sz]
        axs[i].fill_between(input_values, avg - err, avg + err, **_fill_plot_kwargs[sz])
        axs[i].scatter(input_values, evaluated_errors[sz], **_scatter_plot_kwargs[sz])
        axs[i].set_yscale('log')
        axs[i].set_ylabel(f'Sample size {sz}')
    axs[-1].set_xlabel('Parameter ratio')
    if num_sizes == 1:
        axs = axs[0]
    return fig, axs


def plot_precision_test(pvals: Dict[int, List[float]],
                        input_values: List[float],
                        significance: float,
                        fig_axs=None,
                        plot_kwargs: Dict[int, Dict[str, Any]] = None,
                        scatter_kwargs: Dict[int, Dict[str, Any]] = None,
                        plot_all_kwargs: Dict[str, Any] = None,
                        scatter_all_kwargs: Dict[str, Any] = None,
                        **fig_kwargs):
    sizes = list(pvals.keys())
    sizes.sort()
    num_sizes = len(sizes)

    if fig_axs is None:
        fig, axs = plt.subplots(num_sizes, 1, sharex=True, sharey=True, **fig_kwargs)
    else:
        fig, axs = fig_axs
    if num_sizes == 1:
        axs = [axs]

    if plot_kwargs is None:
        plot_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if plot_all_kwargs is None:
        plot_all_kwargs = {}
    if scatter_all_kwargs is None:
        scatter_all_kwargs = {}
    _plot_kwargs = {sz: dict(linestyle='--') for sz in sizes}
    _scatter_kwargs = {sz: dict() for sz in sizes}
    [v.update(plot_all_kwargs) for v in _plot_kwargs.values()]
    [v.update(scatter_all_kwargs) for v in _scatter_kwargs.values()]
    [_plot_kwargs[k].update(v) for k, v in plot_kwargs.items()]
    [_scatter_kwargs[k].update(v) for k, v in scatter_kwargs.items()]

    for i, sz in enumerate(sizes):
        axs[i].plot(input_values, [significance] * len(input_values), **_plot_kwargs[sz])
        axs[i].scatter(input_values, pvals[sz], **_scatter_kwargs[sz])
        axs[i].set_yscale('log')
        axs[i].set_ylabel(f'Sample size {sz}')
    axs[-1].set_xlabel('Parameter ratio')
    if num_sizes == 1:
        axs = axs[0]
    return fig, axs


def save_data(model: Union[SDEModel, Type[SDEModel]],
              metadata: sim_lib.Metadata,
              data_path: str,
              initial_values: Dict[str, float] = None,
              parameters: Dict[str, float] = None,
              **additional_info):
    if not os.path.isdir(os.path.dirname(data_path)):
        os.makedirs(os.path.dirname(data_path))

    data = dict(
        model=model.name,
        metadata=metadata.to_json()
    )
    if initial_values is not None:
        data['initial_values'] = initial_values
    if parameters is not None:
        data['parameters'] = parameters
    if additional_info:
        data['additional_info'] = additional_info
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_data(data_path: str, *model_args, **model_kwargs):
    with open(data_path, 'r') as f:
        data = json.load(f)
    metadata = sim_lib.Metadata.from_json(data['metadata'])
    initial_values = {k: float(v) for k, v in data.get('initial_values', {}).items()}
    parameters = {k: float(v) for k, v in data.get('parameters', {}).items()}
    results = execute_sample(
        metadata.sample_size,
        SDEModel.get(data['model'], *model_args, **model_kwargs),
        metadata.sample_times.shape[0],
        metadata.sample_times[1] - metadata.sample_times[0],
        initial_values,
        parameters
    )
    return metadata, results, initial_values, parameters, data.get('additional_info', {})


def generate_ssr_dataset(model: SDEModel,
                         sample_sizes: List[int],
                         num_steps: int,
                         dt: float,
                         initial_values: Dict[str, float] = None,
                         parameters: Dict[str, float] = None,
                         prefix: str = None,
                         results_dir: str = None,
                         **additional_info):
    metadata_set = {}
    results_repro_set = {}

    storing_data = prefix is not None and results_dir is not None

    for i, sz in enumerate(sample_sizes):
        res_2sz = execute_sample(sz * 2,
                                 model,
                                 num_steps,
                                 dt,
                                 initial_values,
                                 parameters)
        metadata_set[sz] = res_2sz.generate_metadata()
        results_repro_set[sz] = res_2sz.copy()
        results_repro_set[sz].data = results_repro_set[sz].data[:sz, :, :]

        if storing_data:
            save_data(model,
                      metadata_set[sz],
                      os.path.join(results_dir, f'{prefix}_{i}.json'),
                      initial_values,
                      parameters,
                      **additional_info)

    return metadata_set, results_repro_set


def load_ssr_dataset(results_dir: str, prefix: str, *model_args, **model_kwargs):
    metadata_set = []
    results_repro_set = []
    initial_values_set = []
    parameters_set = []
    additional_info_set = []

    for fp in os.listdir(results_dir):
        if fp.startswith(prefix) and fp.endswith('.json'):
            data = load_data(os.path.join(results_dir, fp), *model_args, **model_kwargs)
            metadata, results, initial_values, parameters, additional_info = data
            metadata_set.append(metadata)
            results_repro_set.append(results)
            initial_values_set.append(initial_values)
            parameters_set.append(parameters)
            additional_info_set.append(additional_info)

    return metadata_set, results_repro_set, initial_values_set, parameters_set, additional_info_set


def plot_repro(metadata_set: Dict[int, sim_lib.Metadata],
               fig_kwargs: Dict[str, Any] = None,
               fig_ax=None,
               plot_kwargs: Dict[str, Any] = None,
               n_std: int = 3):
    sample_sizes = list(metadata_set.keys())
    sample_sizes.sort()

    if fig_ax is None:
        if fig_kwargs is None:
            fig_kwargs = {}
        if 'figsize' not in fig_kwargs:
            fig_kwargs['figsize'] = (4.0, 4.0)
        if 'layout' not in fig_kwargs:
            fig_kwargs['layout'] = 'compressed'
        fig, ax = plt.subplots(1, 1, **fig_kwargs)

    else:
        fig, ax = fig_ax

    if plot_kwargs is None:
        plot_kwargs = {}
    if 'marker' not in plot_kwargs:
        plot_kwargs['marker'] = 'o'

    ydata = [metadata_set[sz].ks_stat_mean for sz in sample_sizes]
    yerr = [metadata_set[sz].ks_stat_stdev * n_std for sz in sample_sizes]
    ax.errorbar(sample_sizes, ydata, yerr=yerr, **plot_kwargs)
    if fig_ax is None:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Sample size')
        ax.set_ylabel('EFECT Error')
        ax.set_ylim(min(1E-2, min(ydata) * 0.9), 2.0)

    return fig, ax
