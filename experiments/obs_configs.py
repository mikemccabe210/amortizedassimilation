import torch
from amortized_assimilation.operators import mystery_operator, filter_obs
import numpy
"""
Contains observation configurations for experiments.

Each entry contains a dictionary of input sizes, a dictionary
of observation functions, and a boolean function indicating whether
the observation operator is "known" from the index. 
"""
n = 40

class RandomIndices:
    def __init__(self, n, m=10):
        self.n = n
        self.m = m

    def __getitem__(self, item):
        return filter_obs()

lorenz_configs = {
    # Observe the full state every step
    'full_obs' :   ( {'0': n},
                    {'0': filter_obs(torch.arange(n))},
                    {'0': torch.arange(n)},
                    lambda x: True),
    # Observe every 4th dim each step
    'every_4th_dim_partial_obs' : ({'0': n // 4, '1': n // 4, '2': n // 4, '3': n // 4},
               {'0': filter_obs(torch.arange(n)[0 :n: 4]),
               '1': filter_obs(torch.arange(n)[1 :n: 4]),
               '2': filter_obs(torch.arange(n)[2 :n: 4]),
               '3': filter_obs(torch.arange(n)[3 :n: 4])},
             {'0': torch.arange(n)[0 :n: 4],
               '1': torch.arange(n)[1 :n: 4],
               '2': torch.arange(n)[2 :n: 4],
               '3': torch.arange(n)[3 :n: 4]},
                lambda x: True),
    # 'randomized_obs' : ()
    # Observe either every 4th dim or a randomly projected feature set
    'unstructured_partial_obs': ({'0': n // 4, '1': n // 4, '2': n // 4, '3': n // 4,
                  '4': n//4, '5': n//4, '6': n//4, '7': n//4},
                    {'0': filter_obs(torch.arange(n)[0 :n: 4]),
                        '1': mystery_operator(),
                    '2': filter_obs(torch.arange(n)[1 :n: 4]),
                        '3': mystery_operator(),
                    '4': filter_obs(torch.arange(n)[2 :n: 4]),
                        '5': mystery_operator(),
                    '6' : filter_obs(torch.arange(n)[3 :n: 4]),
                        '7': mystery_operator()},
                         lambda x: x % 2 == 0)
}

vl_n = 36

vl_configs = {
    # Observe the full state every step
    'full_obs' :   ( {'0': vl_n},
                    {'0': filter_obs(torch.arange(vl_n))},
                    {'0': torch.arange(vl_n)},
                    lambda x: True),
    # Observe every 4th dim each step
    'every_4th_dim_partial_obs' : ({'0': vl_n // 4, '1': vl_n // 4, '2': vl_n // 4, '3': vl_n // 4},
               {'0': filter_obs(torch.arange(vl_n)[0 :vl_n: 4]),
               '1': filter_obs(torch.arange(vl_n)[1 :vl_n: 4]),
               '2': filter_obs(torch.arange(vl_n)[2 :vl_n: 4]),
               '3': filter_obs(torch.arange(vl_n)[3 :vl_n: 4])},
             {'0': torch.arange(vl_n)[0 :vl_n: 4],
               '1': torch.arange(vl_n)[1 :vl_n: 4],
               '2': torch.arange(vl_n)[2 :vl_n: 4],
               '3': torch.arange(vl_n)[3 :vl_n: 4]},
                lambda x: True),
    # 'randomized_obs' : ()
    # Observe either every 4th dim or a randomly projected feature set
    'unstructured_partial_obs': ({'0': vl_n // 4, '1': vl_n // 4, '2': vl_n // 4, '3': vl_n // 4,
                  '4': vl_n//4, '5': vl_n//4, '6': vl_n//4, '7': vl_n//4},
                    {'0': filter_obs(torch.arange(vl_n)[0 :vl_n: 4]),
                        '1': mystery_operator(),
                    '2': filter_obs(torch.arange(vl_n)[1 :vl_n: 4]),
                        '3': mystery_operator(),
                    '4': filter_obs(torch.arange(vl_n)[2 :vl_n: 4]),
                        '5': mystery_operator(),
                    '6' : filter_obs(torch.arange(vl_n)[3 :vl_n: 4]),
                        '7': mystery_operator()},
                         lambda x: x % 2 == 0)
}
ks_n = 128
KS_configs = {
    # Observe the full state every step
    'full_obs' :   ( {'0': ks_n},
                    {'0': filter_obs(torch.arange(ks_n))},
                     {'0':torch.arange(ks_n)},
                    lambda x: True),
    'partial_obs': ({'0': 64.,
                     '1': 64.},
                 {'0': filter_obs(torch.arange(ks_n)[0 :ks_n: 2]),
               '1': filter_obs(torch.arange(ks_n)[1 :ks_n: 2])},
                    {'0': torch.arange(ks_n)[0:ks_n: 2],
                     '1': torch.arange(ks_n)[1:ks_n: 2]},
                 lambda x: True),
}