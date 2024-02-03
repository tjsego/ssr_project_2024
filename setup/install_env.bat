@echo off

call conda env remove -n stoch_repro
call conda env create -n stoch_repro -f %~dp0env.yml
