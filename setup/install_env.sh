#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"
source ${HOME}/miniconda3/etc/profile.d/conda.sh

conda env remove -n stoch_repro
conda create -y -n stoch_repro
conda activate stoch_repro
conda install -y python=3.10 scipy matplotlib notebook ipywidgets numba ipympl
pip install antimony libroadrunner