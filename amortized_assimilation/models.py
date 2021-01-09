import torch
import torch.nn.functional as F
import torch.nn as nn

from torchdiffeq import odeint
from .data_utils import L96, L63, etd_rk4_wrapper

class ID_Network(nn.Module):
    """ Returns the input"""
    def forward(self, x, state):
        return x

class EnAF(nn.Module):
    """ Ensemble Amortized Filter 
    
    Used to assimilate a specific observation type. The EnAF trains
    a regression model to predict the next noisy observation using 
    the current noisy observation and an a priori specified set
    of differentiable system dynamics. 
    
    Args
    ----
    obs_size : int
        Size of observation vector
    hidden_size : int
        Size of memory and hidden state for all dense layers. Memory/hidden sized tied 
        implementation convenience, but do not have to be.
    state_size : int
        Size of full system state
    ode_func : nn.Module
        Differentiable numerical model used to push system state forward
    m : int
        Ensemble size
    do : float
        dropout rate
    """
    def __init__(self, obs_size, hidden_size, state_size, ode_func, m, do = .1):
        super(EnAF, self).__init__()
        # Dims
        self.m = m
        self.n_i = obs_size
        self.n_h = hidden_size
        self.n_o = state_size    
        self.ode_func = ode_func
        
        # Linear components
        self.lin_net = nn.Sequential(nn.Linear(obs_size + state_size*2 + hidden_size, hidden_size),
                                     nn.LayerNorm(hidden_size),
                                 nn.LeakyReLU(),
                                     nn.Dropout(do),
                                   nn.Linear(hidden_size, hidden_size),
                                      nn.LayerNorm(hidden_size),
                                 nn.LeakyReLU(),
                                     nn.Dropout(do),
                                   nn.Linear(hidden_size, state_size + hidden_size),
                                   )
        # Sigmoid components                
        self.sig_net = nn.Sequential(nn.Linear(obs_size + state_size*2 + hidden_size, hidden_size),
                                     nn.LayerNorm(hidden_size),
                                 nn.LeakyReLU(),
                                     nn.Dropout(do),
                                   nn.Linear(hidden_size, hidden_size),
                                     nn.LayerNorm(hidden_size),
                                 nn.LeakyReLU(),
                                     nn.Dropout(do),
                                   nn.Linear(hidden_size, state_size + hidden_size),
                                 nn.Sigmoid())
        
    def forward(self, observation, state, memory, tols = (1e-3, 1e-5),
               interval = torch.Tensor([0, .1])):
        """Forward pass for the model
        
        Args
        ----
        observation : torch.Tensor
            Noisy observation vector generated from some function of true state
        state : torch.Tensor
            Current state estimate for system
        memory_states : d
            df
        tols : tuple(float, float)
            Relative and absolute tolerance for adaptive integrators
        interval : torch.Tensor size = (2,)
            Integration time for forward model.
        
        Returns:
            analysis : torch.Tensor
                Analysis point estimate of current system state
            state : torch.Tensor
                Ensemble estimate pushed forward to next time step
            ens : torch.Tensor
                Current ensemble of state estimates
            memory_states : 
                Memory dict
        """
        rtol, atol = tols        
        # Reshape Inputs
        state_var, state_mean = torch.var_mean(state, dim = 1, keepdim = True)
        state = state.reshape(-1, self.n_o ) 
        state_var = state_var.repeat(1, self.m, 1)
        state_var = state_var.reshape(-1, self.n_o )  
        observation = observation.reshape(-1, self.n_i)
        c_in = torch.cat([observation, state, state_var, memory], dim = 1)

        # Linear subnets
        adj = self.lin_net(c_in)
        adj, x_mem = adj.split([self.n_o, self.n_h], 1)   
        # Sigmoid subnets
        filt = self.sig_net(c_in)
        filt, mem_clear = filt.split([self.n_o, self.n_h,], 1)

        # Autoregressive state estimate updates
        memory = mem_clear*memory + x_mem #* filt_mem
        x = filt*state + adj

        # Forward model
        # TODO: This should definitely happen at the beginning of an assimilation
        # cycle instead of the end but went here in an earlier version where it 
        # made more sense. 
        state = odeint(self.ode_func, x, interval, rtol=rtol, atol=atol,
                        method = 'rk4', options = {'step_size': .05})[-1]
        
        # Debugging tool to track forward evals for adaptive integrators
        self.ode_func.fe = 0
        
        # Reshape outputs
        ens = x.view(-1, self.m, self.n_o)
        analysis = ens.mean(dim = 1)
        state = state.view(-1, self.m, self.n_o)
        
        return analysis, state, ens, memory
        
class MultiObs_EnAF(nn.Module):
    """Container for chaining EnAFs for multiple observations.
    
    Args
    ----
    state_size : int
        Size of state vector
    hidden_size : int
        Width of hidden layers and size of memory vector - these
        are tied purely for implementation convenience.
    input_types : dict
        Dictionary of (observation type, observation dimension) pairs. Used to create
        observation-specified assimilation modules. 
    ode_func : nn.Module
        Differentiable numerical model used to push system state forward
    m : int
        Ensemble size
    do : float
        Dropout rate
    """
    def __init__(self, state_size, hidden_size, input_types = {},
                     ode_func = L96(), m = 20, do = .1):
        super(MultiObs_EnAF, self).__init__()
        
        # Use input_types to create appropriately sized sub-networks
        if len(input_types) != 0:
            in_dict = {'default': ID_Network()}
            for k, v in input_types.items():
                label = k
                subnet = EnAF(v, hidden_size, state_size,  ode_func, m = m, do = do)
                in_dict[label] = subnet
            self.input_mods = nn.ModuleDict(in_dict)
        # This path is almost certainly broken by now
        else:
            self.input_mods = nn.ModuleDict({'default': ID_Network()})

    def forward(self, observation, state, memory, obs_type = 'default', 
                verbose = False, tols = (1e-3, 1e-5), interval = torch.Tensor([0, .1])):
        """Forward pass for the model. Chooses appropriate subnet for input type
        and executes forward pass using that subnet. 
        
        Args
        ----
        observation : torch.Tensor
            Noisy observation vector generated from some function of true state
        state : torch.Tensor
            Current state estimate for system
        memory : torch.Tensor
            Current memory state for system
        obs_type : string
            Key indicating source of observation
        tols : tuple(float, float)
            Relative and absolute tolerance for adaptive integrators
        interval : torch.Tensor size = (2,)
            Integration time for forward model.
        
        Returns:
            analysis : torch.Tensor
                Analysis point estimate of current system state
            state : torch.Tensor
                Ensemble estimate pushed forward to next time step
            ens : torch.Tensor
                Analysis ensemble of state estimates
            memory : torch.Tensor
                Post-update memory state
        """
        # Get appropriate subnetwork for obs_type and assimilate
        H = self.input_mods[obs_type]
        analysis, state, ens, memory = H(observation, state, memory, tols, interval)
        return analysis, state, ens, memory

class ConvEnAF(nn.Module):
    """ EnAF using convolutional base networks
    
    Used to assimilate a specific observation type. The ConvEnAF trains
    a regression model to predict the next noisy observation using 
    the current noisy observation and an a priori specified set
    of differentiable system dynamics. 
    
    Args
    ----
    obs_size : int
        Size of observation vector
    hidden_size : int
        Size of memory and hidden state for all dense layers. Memory/hidden sized tied 
        implementation convenience, but do not have to be.
    state_size : int
        Size of full system state
    ode_func : nn.Module
        Differentiable numerical model used to push system state forward
    m : int
        Ensemble size
    do : float
        dropout rate
    """
    def __init__(self, obs_size, hidden_size, state_size, ode_func, m, do = .1, missing = False):
        super(ConvEnAF, self).__init__()
        # Dims
        self.m = m
        self.n_i = obs_size
        self.n_h = hidden_size
        self.n_o = state_size    
        self.ode_func = ode_func
        
        # Linear components

        if missing:
            n_in = 11
        else:
            n_in = 10
        self.base_net = nn.Sequential(nn.Conv1d(n_in, hidden_size, kernel_size = 5,
                                         padding = 2, padding_mode = 'circular',
                                         groups = 1),
                                     nn.LayerNorm(40),
                                    nn.LeakyReLU(),
                                     SpatialDropout1d(do),
                                    nn.Conv1d(hidden_size, hidden_size, kernel_size = 5,
                                         padding = 2,  dilation = 1, padding_mode = 'circular',
                                         groups = 1),
                                      nn.LayerNorm(40),
                                 nn.LeakyReLU(),
                                     SpatialDropout1d(do),
                                   nn.Conv1d(hidden_size, hidden_size, kernel_size = 5,
                                         padding = 2, dilation = 1,  padding_mode = 'circular',
                                         groups = 1),
                                      nn.LayerNorm(40),
                                 nn.LeakyReLU(),
                                      SpatialDropout1d(do),
                                      nn.Conv1d(hidden_size, 14, kernel_size=5,
                                                padding=2, dilation=1, padding_mode='circular',
                                                groups=1)
                                   )
        # self.lin_net = nn.Sequential(SpatialDropout1d(do),
        #                                 nn.Conv1d(hidden_size, 7, kernel_size = 5,
        #                                  padding = 2, dilation = 1, padding_mode = 'circular',
        #                                  groups = 1),)
        # # Sigmoid components
        # self.sig_net = nn.Sequential(SpatialDropout1d(do),
        #                                 nn.Conv1d(hidden_size, 7, kernel_size = 5,
        #                                  padding = 2, dilation = 1, padding_mode = 'circular',
        #                                  groups = 1), nn.Sigmoid())

        # self.sig_net = nn.Sequential(nn.Conv1d(n_in, hidden_size, kernel_size = 5,
        #                                  padding = 2, padding_mode = 'circular',
        #                                  groups = 1),
        #                              nn.LayerNorm(40),
        #                             nn.LeakyReLU(),
        #                              # SpatialDropout1d(do),
        #                             nn.Conv1d(hidden_size, hidden_size, kernel_size = 5,
        #                                  padding = 2, dilation = 1,  padding_mode = 'circular',
        #                                  groups = 1),
        #                               nn.LayerNorm(40),
        #                          nn.LeakyReLU(),
        #                              # SpatialDropout1d(do),
        #                                nn.Conv1d(hidden_size, hidden_size, kernel_size = 5,
        #                                  padding = 2, dilation = 1,  padding_mode = 'circular',
        #                                  groups = 1),
        #                               nn.LayerNorm(40),
        #                          nn.LeakyReLU(),
        #                              # SpatialDropout1d(do),
        #                         nn.Conv1d(hidden_size, 7, kernel_size = 5,
        #                                  padding = 2, dilation = 1, padding_mode = 'circular',
        #                                  groups = 1),
        #                          nn.Sigmoid())
        
        
    def forward(self, observation, state, memory, mask = None, tols = (1e-3, 1e-5),
               interval = torch.Tensor([0, .1])):
        """Forward pass for the model
        
        Args
        ----
        observation : torch.Tensor
            Noisy observation vector generated from some function of true state
        state : torch.Tensor
            Current state estimate for system
        memory_states : d
            df
        tols : tuple(float, float)
            Relative and absolute tolerance for adaptive integrators
        interval : torch.Tensor size = (2,)
            Integration time for forward model.
        
        Returns:
            analysis : torch.Tensor
                Analysis point estimate of current system state
            state : torch.Tensor
                Ensemble estimate pushed forward to next time step
            ens : torch.Tensor
                Current ensemble of state estimates
            memory_states : 
                Memory dict
        """
        rtol, atol = tols        
        # Reshape Inputs
        state_var, state_mean = torch.var_mean(state, dim = 1, keepdim = True)
        state_var = state_var.repeat(1, self.m, 1)
        state_var = state_var.reshape(-1, self.n_o )  
        state = state.reshape(-1, self.n_o )
        ddt = self.ode_func(0, state)
        observation = observation.reshape(-1, self.n_o)
        if mask is None:
            c_in = torch.stack([observation, state, ddt, state_var], dim = 1)
        else:
            mask = mask.reshape(-1, self.n_o)
            c_in = torch.stack([observation, state, ddt, state_var, mask], dim = 1)
        c_in = torch.cat([c_in, memory], dim = 1)
        # print(c_in.shape)

        base = self.base_net(c_in).squeeze()
        # Linear subnets
        # adj = self.lin_net(base).squeeze()
        adj, x_mem, filt, mem_clear = base.split([1, 6, 1, 6], 1)
        # Sigmoid subnets
        # filt = self.sig_net(base).squeeze()
        # filt, mem_clear = filt.split([1, 6], 1)
        # Autoregressive state estimate updates
        memory = torch.sigmoid(mem_clear)*memory + x_mem #* filt_mem
        x = torch.sigmoid(filt.squeeze())*state + adj.squeeze()
        # Forward model
        # TODO: This should definitely happen at the beginning of an assimilation
        # cycle instead of the end but went here in an earlier version where it 
        # made more sense. 
        state = odeint(self.ode_func, x, interval, rtol=rtol, atol=atol,
                        method = 'rk4', options = {'step_size': .05})[-1]
        
        # Debugging tool to track forward evals for adaptive integrators
        self.ode_func.fe = 0
        
        # Reshape outputs
        ens = x.view(-1, self.m, self.n_o)
        analysis = ens.mean(dim = 1)
        state = state.view(-1, self.m, self.n_o)
        
        return analysis, state, ens, memory
        
class MultiObs_ConvEnAF(nn.Module):
    """Container for chaining ConvEnAFs for multiple observations.
    
    Args
    ----
    state_size : int
        Size of state vector
    hidden_size : int
        Width of hidden layers and size of memory vector - these
        are tied purely for implementation convenience.
    input_types : dict
        Dictionary of (observation type, observation dimension) pairs. Used to create
        observation-specified assimilation modules. 
    ode_func : nn.Module
        Differentiable numerical model used to push system state forward
    m : int
        Ensemble size
    do : float
        Dropout rate
    """
    def __init__(self, state_size, hidden_size, input_types = {},
                     ode_func = L96(), m = 20, do = .1, missing = False):
        super(MultiObs_ConvEnAF, self).__init__()
        self.missing = missing
        # Use input_types to create appropriately sized sub-networks
        if len(input_types) != 0:
            in_dict = {'default': ID_Network()}
            for k, v in input_types.items():
                label = k
                subnet = ConvEnAF(v, hidden_size, state_size, ode_func, m = m, missing = missing, do = do)
                in_dict[label] = subnet
            self.input_mods = nn.ModuleDict(in_dict)
        # This path is almost certainly broken by now
        else:
            self.input_mods = nn.ModuleDict({'default': ID_Network()})

    def forward(self, observation, state, memory, mask = None, obs_type = 'default', 
                verbose = False, tols = (1e-3, 1e-5), interval = torch.Tensor([0, .1])):
        """Forward pass for the model. Chooses appropriate subnet for input type
        and executes forward pass using that subnet. 
        
        Args
        ----
        observation : torch.Tensor
            Noisy observation vector generated from some function of true state
        state : torch.Tensor
            Current state estimate for system
        memory : torch.Tensor
            Current memory state for system
        obs_type : string
            Key indicating source of observation
        tols : tuple(float, float)
            Relative and absolute tolerance for adaptive integrators
        interval : torch.Tensor size = (2,)
            Integration time for forward model.
        
        Returns:
            analysis : torch.Tensor
                Analysis point estimate of current system state
            state : torch.Tensor
                Ensemble estimate pushed forward to next time step
            ens : torch.Tensor
                Analysis ensemble of state estimates
            memory : torch.Tensor
                Post-update memory state
        """
        # Get appropriate subnetwork for obs_type and assimilate
        H = self.input_mods[obs_type]
        analysis, state, ens, memory = H(observation, state, memory, mask, tols, interval)
        return analysis, state, ens, memory
    
class SpatialDropout1d(nn.Module):
    def __init__(self, p):
        super(SpatialDropout1d, self).__init__()
        self.p = p
        self.do = nn.Dropout2d(p)
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.do(x)
        return x.squeeze(2)
        
class KSConvEnAF(nn.Module):
    """ Since the KS integrator is slightly hacky, it was easier 
    to just c+p the ConvEnAF and replace integration. Need to 
    get rid of this once KS is standardized.
    
    Used to assimilate a specific observation type. The EnAF trains
    a regression model to predict the next noisy observation using 
    the current noisy observation and an a priori specified set
    of differentiable system dynamics. 
    
    Args
    ----
    obs_size : int
        Size of observation vector
    hidden_size : int
        Size of memory and hidden state for all dense layers. Memory/hidden sized tied 
        implementation convenience, but do not have to be.
    state_size : int
        Size of full system state
    ode_func : nn.Module
        Differentiable numerical model used to push system state forward
    m : int
        Ensemble size
    do : float
        dropout rate
    """
    def __init__(self, obs_size, hidden_size, state_size, ode_func, m, do = .1, device = None):
        super(KSConvEnAF, self).__init__()
        # Dims
        self.m = m
        self.n_i = obs_size
        self.n_h = hidden_size
        self.n_o = state_size    
        self.ode_func = etd_rk4_wrapper(device)
        n_in = 7
        # Linear components
        self.lin_net = nn.Sequential(nn.Conv1d(n_in, hidden_size, kernel_size = 5,
                                         padding = 2, padding_mode = 'zeros',
                                         groups = 1),
                                     nn.LayerNorm(128),
                                    nn.LeakyReLU(),
                                     SpatialDropout1d(do),
                                    nn.Conv1d(hidden_size, hidden_size, kernel_size = 5,
                                         padding = 2,  dilation = 1, padding_mode = 'zeros',
                                         groups = 1),
                                      nn.LayerNorm(128),
                                 nn.LeakyReLU(),
                                     SpatialDropout1d(do),
                                   nn.Conv1d(hidden_size, hidden_size, kernel_size = 5,
                                         padding = 2, dilation = 1,  padding_mode = 'zeros',
                                         groups = 1),
                                      nn.LayerNorm(128),
                                 nn.LeakyReLU(),
                                     SpatialDropout1d(do),
                                nn.Conv1d(hidden_size, 5, kernel_size = 5,
                                         padding = 2, dilation = 1, padding_mode = 'zeros',
                                         groups = 1),
                                   )
        # Sigmoid components                
        self.sig_net = nn.Sequential(nn.Conv1d(n_in, hidden_size, kernel_size = 5,
                                         padding = 2, padding_mode = 'zeros',
                                         groups = 1),
                                     nn.LayerNorm(128),
                                    nn.LeakyReLU(),
                                    SpatialDropout1d(do),
                                    nn.Conv1d(hidden_size, hidden_size, kernel_size = 5,
                                         padding = 2, dilation = 1,  padding_mode = 'zeros',
                                         groups = 1),
                                      nn.LayerNorm(128),
                                 nn.LeakyReLU(),
                                     SpatialDropout1d(do),
                                       nn.Conv1d(hidden_size, hidden_size, kernel_size = 5,
                                         padding = 2, dilation = 1,  padding_mode = 'zeros',
                                         groups = 1),
                                      nn.LayerNorm(128),
                                 nn.LeakyReLU(),
                                     SpatialDropout1d(do),
                                nn.Conv1d(hidden_size, 5, kernel_size = 5,
                                         padding = 2, dilation = 1, padding_mode = 'zeros',
                                         groups = 1),
                                 nn.Sigmoid())
        
        
    def forward(self, observation, state, memory, tols = (1e-3, 1e-5),
               interval = torch.Tensor([0, .1])):
        """Forward pass for the model
        
        Args
        ----
        observation : torch.Tensor
            Noisy observation vector generated from some function of true state
        state : torch.Tensor
            Current state estimate for system
        memory_states : d
            df
        tols : tuple(float, float)
            Relative and absolute tolerance for adaptive integrators
        interval : torch.Tensor size = (2,)
            Integration time for forward model.
        
        Returns:
            analysis : torch.Tensor
                Analysis point estimate of current system state
            state : torch.Tensor
                Ensemble estimate pushed forward to next time step
            ens : torch.Tensor
                Current ensemble of state estimates
            memory_states : 
                Memory dict
        """
        rtol, atol = tols        
        # Reshape Inputs
        state_var, state_mean = torch.var_mean(state, dim = 1, keepdim = True)
        state_var = state_var.repeat(1, self.m, 1)
        state_var = state_var.reshape(-1, self.n_o) 
        state = state.reshape(-1, self.n_o )   
        observation = observation.reshape(-1, self.n_i)
        
        c_in = torch.stack([observation, state, state_var], dim = 1)
        c_in = torch.cat([c_in, memory], dim = 1)
        # print(c_in.shape)

        # Linear subnets
        adj = self.lin_net(c_in).squeeze()
        # print(adj.shape)
        adj, x_mem = adj.split([1, 4], 1)   
        # Sigmoid subnets
        filt = self.sig_net(c_in).squeeze()
        filt, mem_clear = filt.split([1, 4], 1)

        # Autoregressive state estimate updates
        memory = mem_clear*memory + x_mem #* filt_mem
        x = filt.squeeze()*state + adj.squeeze()

        # Forward model
        # TODO: This should definitely happen at the beginning of an assimilation
        # cycle instead of the end but went here in an earlier version where it 
        # made more sense. 
#         state = odeint(self.ode_func, x, interval, rtol=rtol, atol=atol,
#                         method = 'rk4', options = {'step_size': .05})[-1]
        state = self.ode_func(x, .5, .5, True)
        state = self.ode_func(state, .5, .5, True)
        
        # Debugging tool to track forward evals for adaptive integrators
#         self.ode_func.fe = 0
        
        # Reshape outputs
        ens = x.view(-1, self.m, self.n_o)
        analysis = ens.mean(dim = 1)
        state = state.view(-1, self.m, self.n_o)
        
        return analysis, state, ens, memory
        
class MultiObs_KSConvEnAF(nn.Module):
    """Container for chaining ConvEnAFs for multiple observations.
    
    Args
    ----
    state_size : int
        Size of state vector
    hidden_size : int
        Width of hidden layers and size of memory vector - these
        are tied purely for implementation convenience.
    input_types : dict
        Dictionary of (observation type, observation dimension) pairs. Used to create
        observation-specified assimilation modules. 
    ode_func : nn.Module
        Differentiable numerical model used to push system state forward
    m : int
        Ensemble size
    do : float
        Dropout rate
    """
    def __init__(self, state_size, hidden_size, input_types = {},
                     ode_func = L96(), m = 20, do = .1, device = None):
        super(MultiObs_KSConvEnAF, self).__init__()
        
        # Use input_types to create appropriately sized sub-networks
        if len(input_types) != 0:
            in_dict = {'default': ID_Network()}
            for k, v in input_types.items():
                label = k
                subnet = KSConvEnAF(v, hidden_size, state_size, ode_func, m = m, device = device)
                in_dict[label] = subnet
            self.input_mods = nn.ModuleDict(in_dict)
        # This path is almost certainly broken by now
        else:
            self.input_mods = nn.ModuleDict({'default': ID_Network()})

    def forward(self, observation, state, memory, obs_type = 'default', 
                verbose = False, tols = (1e-3, 1e-5), interval = torch.Tensor([0, .1])):
        """Forward pass for the model. Chooses appropriate subnet for input type
        and executes forward pass using that subnet. 
        
        Args
        ----
        observation : torch.Tensor
            Noisy observation vector generated from some function of true state
        state : torch.Tensor
            Current state estimate for system
        memory : torch.Tensor
            Current memory state for system
        obs_type : string
            Key indicating source of observation
        tols : tuple(float, float)
            Relative and absolute tolerance for adaptive integrators
        interval : torch.Tensor size = (2,)
            Integration time for forward model.
        
        Returns:
            analysis : torch.Tensor
                Analysis point estimate of current system state
            state : torch.Tensor
                Ensemble estimate pushed forward to next time step
            ens : torch.Tensor
                Analysis ensemble of state estimates
            memory : torch.Tensor
                Post-update memory state
        """
        # Get appropriate subnetwork for obs_type and assimilate
        H = self.input_mods[obs_type]
        analysis, state, ens, memory = H(observation, state, memory, tols, interval)
        return analysis, state, ens, memory