# Amortized Assimilation

This repository contains a PyTorch implementation of the paper *Learning to Assimilate in Chaotic Dynamical
Systems*. 

Abstract: The accuracy of simulation-based forecasting in chaotic systems is heavily dependent on 
high-quality estimates of the system state at the time the forecast is initialized. Data assimilation 
methods are used to infer these initial conditions by systematically combining noisy, incomplete 
observations and numerical models of system dynamics to produce effective estimation schemes. We 
introduce \textit{amortized assimilation}, a framework for learning to assimilate in dynamical 
systems from sequences of noisy observations with no need for ground truth data. We motivate the 
framework by extending powerful results from self-supervised denoising to the dynamical systems 
setting through the use of differentiable simulation. Experimental results across several benchmark 
systems highlight the improved effectiveness of our approach over widely-used data assimilation 
methods.
## Installation

### Requirements

This code can be memory heavy as each experiment unrolls at least 40 assimilation steps (which from a memory 
perspective is equivalent to a 40x deeper network plus whatever is needed for the simulation). Current settings are 
optimized to max out memory usage on a GTX1070 GPU. The easiest ways to tune memory usage are network width and ensemble 
size. Checkpointing could significantly improve memory utilization but is not currently implemented.

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
All experiments can be run from experiments/run_*.py. Default settings are those used in the paper.
First navigate to the experiments directory then execute:

L96 Full Observations
```
python run_L96Conv.py --obs_conf full_obs
```
L96 Partial Observations (every fourth). 
```
python run_L96Conv.py --obs_conf every_4th_dim_partial_obs
```
VL20 Partial 
```
python run_VLConv.py --obs_conf every_4th_dim_partial_obs
```
KS Full
```
python run_KS.py 
```

Other modifications of interest might be to adjust the step size for the integrator (--step_size, default .1), 
observation error(--noise, default 1.), ensemble size (--m, default 10), or 
network width (--hidden_size, default 64 for conv). The L96 code also includes options for self-supervised 
and supervised analysis losses (ss_analysis, clean_analysis) used for creating Figure 6 from the paper. Custom observation 
operators can be created in the same style as those found in obs_configs.py. 
 
Parameters for traditional DA approaches were tuned via grid search over smaller sequences. Those hyperparameters were 
then used for longer assimilation sequences.

To test a new architecture, you'll want to ensure it's obeying the same API as the models in models.py, but otherwise
it should slot in without major issues.

## Datasets
Code is included for generating the Lorenz 96, VL 20 and KS datasets. This can be found under amortized_assimilation/data_utils.py
## References

DAPPER: Raanes, P. N., & others. (2018). nansencenter/DAPPER: Version 0.8. https://doi.org/10.5281/zenodo.2029296

## Acknowledgements

This material is based upon work supported by the National Science Foundation under Grant No. 1835825. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
---

If you found the code or ideas in this repository useful, please consider citing:
```
@article{mccabe2021l2assim,
  title={Learning to Assimilate in Chaotic Dynamical Systems},
  author={McCabe, Michael and Brown, Jed},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```
