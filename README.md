# Amortized Assimilation

This respository contains a PyTorch implementation of the (WIP) paper *Learning to Assimilate in Chaotic Dynamical
Systems*. This work was initially presented at the *NeurIPS 2020 Workshop on Differentiable Vision, Graphics, and Physics in
Machine Learning*.

Abstract: The accuracy of simulation-based forecasting in chaotic systems, such as those found in weather and climate settings, is
heavily dependent on the accuracy of the initial condition estimates used in the forward simulation. Data assimilation 
methods seek to infer these initial conditions by systematically combining both noisy observations and numerical models 
of system dynamics to produce highly effective estimation schemes. In this work, we introduce a self-supervised 
framework for learning how to assimilate a given dynamical system from data which we call Amortized Assimilation. Our 
approach combines deep learning-based denoising with differentiable simulation, using independent neural networks 
to assimilate specific observation types while connecting the gradient flow between these sub-tasks with 
differentiable simulation and shared memory. We develop a training objective for this hybrid architecture which is 
provably minimized by an unbiased estimator of the true system state despite training with only noisy data and 
demonstrate through numerical experiments that amortized assimilators outperform widely used data assimilation 
methods across several standard benchmarks.
## Installation

### Requirements

This code can be memory heavy as each experiment unrolls at least 40 assimilation steps (which from a memory 
perspective is equivalent to a 40x deeper network plus whatever is needed for the simulation). Current settings are 
optimized to max out memory usage on a GTX1070 GPU. The easiest ways to tune memory usage are network width and ensemble 
size.

To install the dependencies, use the provided requirements.txt file:
```
pip install -r requirements.txt 
```
There is also a dependency on torchdiffeq. Instructions for installing torchdiffeq can be found at 
https://github.com/rtqichen/torchdiffeq, but are also copied below:
```
pip install git+https://github.com/rtqichen/torchdiffeq
```
To run the DA comparison models, you will need to install DAPPER. Instructions can be found here: 
https://github.com/nansencenter/DAPPER.
### Installing this package

A setup.py file has been included for installation. Navigate to the home folder and run:

```
pip install -e . 
```

## Run experiments
All experiments can be run from experiments/run_*.py. First navigate to the experiments directory then execute:

L96 Full Observations
```
python run_L96FF.py --obs_conf full_obs
python run_L96Conv.py --obs_conf full_obs
```
L96 Partial Observations (every fourth). 
```
python run_L96FF.py --obs_conf every_4th_dim_partial_obs
python run_L96Conv.py --obs_conf every_4th_dim_partial_obs
```
KS Full
```
python run_KS.py 
```
Misspecified Model
```
python run_misspec.py 
```
Other modifications of interest might be to adjust the step size (--step_size, default .1), 
observation error(--noise, default 1.), ensemble size (--m, default 10), or 
network width (--hidden_size, default 250 for FF, 64 for conv). Custom observation configs can be created in the same 
style as those found in obs_configs.py. 

Parameters for traditional DA approaches were tuned via grid search over smaller sequences. The code for that search can
be found under experiments/dapper_experiments and results were piped into text files also stored in that directory. 
Those hyperparameters were then used for longer assimilation sequences. 

## Datasets
Code is included for generating the Lorenz 96 and KS datasets. The Lorenz 2005 data was generated using the DAPPER 
library as there was no need to differentiate through them. 
## References

DAPPER: Raanes, P. N., & others. (2018). nansencenter/DAPPER: Version 0.8. https://doi.org/10.5281/zenodo.2029296