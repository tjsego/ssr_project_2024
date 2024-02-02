#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" ; pwd -P )"
source ${HOME}/miniconda3/etc/profile.d/conda.sh

conda env remove -n stoch_repro