@echo off

call conda env remove -n stoch_repro
call conda create -y -n stoch_repro
call conda activate stoch_repro
call conda install -y python=3.10 scipy matplotlib notebook ipywidgets
pip install antimony libroadrunner
