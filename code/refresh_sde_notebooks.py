import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil

this_dir = os.path.dirname(os.path.abspath(__file__))
notebook_output = os.path.join(this_dir, 'refreshed_sde_notebooks')

notebook_dir = os.path.join(os.path.dirname(this_dir), 'notebooks')

notebooks = [
    'pandemic_adak2020',
    'pandemic_din2020',
    'pandemic_faranda2020',
    'pandemic_mamis2023',
    'pandemic_ninotorres2022',
    'pandemic_tesfaye2020'
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
