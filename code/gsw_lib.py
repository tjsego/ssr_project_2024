import multiprocessing as mp
import numpy as np
import os
from typing import List, Optional, Tuple

import sim_lib


def _load_data(label_offset: int, num_entries: int, f_path: str, prefices) -> Optional[Tuple[int, str, np.ndarray]]:
    res_label = -1
    prefix = ''
    f_path_basename = os.path.basename(f_path)
    for p in prefices:
        if p in f_path_basename:
            prefix = p
            res_label = int(f_path_basename.replace(p, '').replace('.txt', '')) - label_offset

    if num_entries > 0:
        if res_label >= num_entries:
            return None
    
    d = []
    with open(f_path, 'r') as fo:
        for s in fo:
            d.append(float(s))

    return res_label, prefix, np.asarray(d, dtype=float)


def load_data(prefices: List[str], res_dir: str, label_offset: int = 0, num_entries=-1):
    res_file_paths = [os.path.join(os.path.abspath(res_dir), f) for f in os.listdir(res_dir)]
    num_workers = min(len(res_file_paths), mp.cpu_count())
    
    obj_data = {p: dict() for p in prefices}
    num_steps = None
    
    with mp.Pool(num_workers) as p:
        for pool_data in p.starmap(_load_data, [(label_offset, num_entries, f_path, prefices) for f_path in res_file_paths]):
            if pool_data is None:
                continue
            res_label, prefix, prefix_data = pool_data
            obj_data[prefix][res_label] = prefix_data
            if num_steps is None:
                num_steps = prefix_data.shape[0]

    rep_nums = list(obj_data[prefices[0]].keys())
    rep_nums.sort()
    num_reps = len(rep_nums)

    fin_data = {p: np.ndarray((num_reps, num_steps), dtype=float) for p in prefices}
    for p in prefices:
        for r in rep_nums:
            fin_data[p][r, :] = obj_data[p][r]

    return np.asarray(list(range(num_steps)), dtype=float), fin_data


def _compare_experiment_implementations(sz: int, 
                                        param: str,
                                        eval_num: int, 
                                        eval_fin: float,
                                        data1: np.ndarray,
                                        data2: np.ndarray):
    eval_t = sim_lib.get_eval_info_times((eval_num, eval_fin))
    kss = sim_lib.ecf_compare(sim_lib.ecf(data1, eval_t), sim_lib.ecf(data2, eval_t))
    return sz, param, kss


def compare_experiment_implementations(prefices: List[str], data_impl1, data_impl2, eval_info1, eval_info2):
    _kss_size = {sz: dict() for sz in data_impl1.keys()}

    input_args = []
    for sz in eval_info1.keys():
        _kss_size[sz] = {p: -1.0 for p in prefices}
        for p in prefices:
            for time_idx in range(len(eval_info1[sz][p])):
                input_args.append((
                    sz,
                    p,
                    max(eval_info1[sz][p][time_idx][0], eval_info2[sz][p][time_idx][0]),
                    max(eval_info1[sz][p][time_idx][1], eval_info2[sz][p][time_idx][1]),
                    data_impl1[p][:sz, time_idx].T,
                    data_impl2[p][:sz, time_idx].T
                ))

    num_workers = min(mp.cpu_count(), len(input_args))
    with mp.Pool(num_workers) as p:
        for sz, param, kss in p.starmap(_compare_experiment_implementations, input_args):
            _kss_size[sz][param] = max(_kss_size[sz][param], kss)

    return _kss_size
