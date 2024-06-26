import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil

this_dir = os.path.dirname(os.path.abspath(__file__))
notebook_output = os.path.join(this_dir, 'refreshed_notebooks')

notebook_dir = os.path.join(os.path.dirname(this_dir), 'notebooks')

notebooks = [
    'gsw',
    'proto_10paramvar',
    'proto_10paramvar2',
    'proto_2paramvar',
    'proto_bistable',
    'proto_bistable2',
    'proto_bm_banerjee2008',
    'proto_bm_liu2012',
    'proto_bm_lo2005',
    'proto_compare_0',
    'proto_compare_1',
    'proto_compare_2',
    'proto_compare_3',
    'proto_compare_4',
    'proto_compare_5',
    'proto_compare_5a',
    'proto_compare_6',
    'proto_compare_7',
    'proto_compare_7a',
    'proto_compare_7b',
    'proto_compare_boolean',
    'proto_compare_mag_1',
    'proto_compare_periods_1',
    'proto_compare_res_1',
    'proto_compare_size_1',
    'proto_compare_var_1',
    'proto_compare_var_2',
    'proto_lorentz',
    'proto_oscillator',
    'proto_paramvar',
    'proto_pendulum',
    'proto_pulse',
    'proto_seir',
    'sim_modeler_1',
    'sim_curator_1_fail',
    'sim_curator_1_pass',
    'sim_modeler_2',
    'sim_curator_2_fail_params',
    'sim_curator_2_fail_sigfig',
    'sim_curator_2_pass'
]

if __name__ == '__main__':

    if os.path.isdir(notebook_output):
        shutil.rmtree(notebook_output)
    os.mkdir(notebook_output)

    ep = ExecutePreprocessor()

    for name in notebooks:
        print('Working notebook:', name)

        notebook_path_rel = name + '.ipynb'
        notebook_path_old = os.path.join(notebook_dir, notebook_path_rel)
        with open(notebook_path_old) as f:
            nb = nbformat.read(f, as_version=4)
        ep.preprocess(nb, {'metadata': {'path': this_dir}})

        notebook_path_new = os.path.join(notebook_output, notebook_path_rel)
        with open(notebook_path_new, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
