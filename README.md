# Amortized Assimilation

This respository contains a PyTorch implementation of the paper *Self-supervised Assimilation of Chaotic Dynamical Systems*.

Inferring the state of chaotic dynamical systems is a core component of forecasting in the geosciences. Modern data assimilation methods combine noisy observations with physics-derived dynamical models to produce more accurate state estimates than can be obtained from either source alone. Amortized assimilation is a deep learning based data assimilation framework for estimating a nonparametric approximation to the posterior state density. Amortized assimilators extend recurrent networks with physics-based system dynamics, facilitating end-to-end gradient-based training. These hybrid models are able to learn to assimilate complex input distributions while maintaining an easily computable test-time update step. 

## Installation

### Requirements

To install the dependencies, use the provided requirements.txt file:
```
pip install -r requirements.txt 
```
There is also a dependencies on torchdiffeq. Instructions for installing torchdiffeq can be found at https://github.com/rtqichen/torchdiffeq, but are also copied below:
```
pip install git+https://github.com/rtqichen/torchdiffeq
```
To run the DA comparison models, you will need to install DAPPER. Instructions can be found here: https://github.com/nansencenter/DAPPER.
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
Other modifications of interest might be to adjust the step size (--step_size, default .1), observation error(--noise, default 1.), ensemble size (--m, default 10), or network width (--hidden_size, default 250 for FF, 32 for conv). Custom observation configs can be created in the same style as those found in obs_configs.py. 

## Datasets
Code is included for generating the L40 and KS datasets. Two level lorenz was generated using the DAPPER library as there was no need to differentiate through them.
## References

torchdiffeq: Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. Advances in Neural Information Processing Systems. https://doi.org/10.2307/j.ctvcm4h3p.19

DAPPER: Raanes, P. N., & others. (2018). nansencenter/DAPPER: Version 0.8. https://doi.org/10.5281/zenodo.2029296