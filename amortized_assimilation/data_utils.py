import torch
import torch.nn as nn
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint


class L96(nn.Module):
    def __init__(self):
        super(L96, self).__init__()
        self.fe = 0
    def forward(self, t, x, F = 8.):
        self.fe += 1
        x_m2 = torch.roll(x, -2, -1)
        x_m1 = torch.roll(x, -1, -1)
        x_p1 = torch.roll(x, 1, -1)
        return (x_p1 - x_m2) * x_m1 - x + F * torch.ones_like(x)

class VL20(nn.Module):
    """Modeled after dapper implementation"""
    def __init__(self, nX=36, F=10, G=10, alpha=1, gamma=1):
        super(VL20, self).__init__()
        self.fe = 0
        self.nX = nX
        self.F = F
        self.G = G
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, t, x):
        self.fe += 1
        out = torch.zeros_like(x)
        # print( torch.split(x, self.nX, -1))
        X, theta = x[:, 0, :], x[:, 1, :]

        # Velocities
        out[:, 0, :] = (torch.roll(X, 1, -1) - torch.roll(X, -2, -1))*torch.roll(X, -1, -1)
        out[:, 0, :] -= self.gamma*X
        out[:, 0, :] += self.F - self.alpha * theta
        # Temperatures
        out[:, 1, :] = torch.roll(X, 1, -1)*torch.roll(theta, 2, -1) - \
                           torch.roll(X, -1, -1) * torch.roll(theta, -2, -1)
        out[:, 1, :] -= self.gamma*theta
        out[:, 1, :] += self.alpha*X + self.G
        return out


class L63(nn.Module):
    def __init__(self):
        super(L63, self).__init__()
        self.fe = 0
    def forward(self, t, x, sig = 10.0, rho = 28.0, beta = 8.0/3):
        rvals = torch.zeros_like(x)
        rvals[:, 0] = sig * ( x[:, 1] - x[:, 0] )
        rvals[:, 1] = x[:, 0] * (rho - x[:, 2]) - x[:, 1]
        rvals[:, 2] = x[:, 0] * x[:, 1] - beta * x[:, 2]
        return rvals
    
def gen_data(dataset, t, steps_test, steps_valid, step = None, check_disk = True, steps_burn = 1000):
    """ Generates training and test data for given model. Defaults used in experiments
    are hardcoded making this somewhat longer than it needs to be.

    args
    -----
    dataset : string
        Name of the dynamics to use for data gen
    steps_test : int
        Number of steps to use for test set. Steps_test + steps_valid Must be smaller than len(t)
    steps_valid : int
        Number of steps to use for valid set. Steps_test + steps_valid Must be smaller than len(t)
    step : int
        Step size taken by integrator. If no step size provided, uses t[1] - t[0]
    check_disk : bool
        Indicates whether to check if saved first.
    
    returns
    -------
    train : torch.Tensor
        Training sequence
    valid : torch.Tensor
        Validation sequence
    test : torch.Tensor
        Test sequence
    """
    makedirs('data/%s' % dataset)
    if step is None:
        step = t[1] - t[0]
    rstep =  t[1] - t[0]
    if dataset == 'lorenz96':
        with torch.no_grad():
            if check_disk and os.path.exists('data/%s/true_y_%.3fstep.npy' % (dataset, rstep)):
                true_y = torch.Tensor(np.load('data/%s/true_y_%.3fstep.npy' % (dataset, rstep)))         
                true_y0_test = true_y[-1]
                true_y = true_y[:-1]
                true_y_test = odeint(L96(), true_y0_test, t[:steps_burn + steps_valid + steps_test],
                                    method='rk4', options = {'step_size': step} )  
            else:
                true_y0 = torch.randn(1, 40) + 5
                true_y = odeint(L96(), true_y0, t, method='rk4', options = {'step_size': step} )
                if check_disk:
                    np.save('data/%s/true_y_%.3fstep.npy' % (dataset, rstep), true_y)
                true_y0_test = true_y[-1]
                true_y = true_y[:-1]
                true_y_test = odeint(L96(), true_y0_test, t[:steps_burn + steps_test + steps_valid],
                            method='rk4', options = {'step_size': step} )
    elif dataset == 'vl20':
        with torch.no_grad():
            if check_disk and os.path.exists('data/%s/true_y_%.3fstep.npy' % (dataset, rstep)):
                true_y = torch.Tensor(np.load('data/%s/true_y_%.3fstep.npy' % (dataset, rstep)))
                true_y0_test = true_y[-1]
                true_y = true_y[:-1]
                true_y_test = odeint(VL20(), true_y0_test, t[:steps_burn + steps_valid + steps_test],
                                     method='rk4', options={'step_size': step})
            else:
                true_y0 = torch.randn(1, 2, 36) + 5
                true_y = odeint(VL20(), true_y0, t, method='rk4', options={'step_size': step})
                if check_disk:
                    np.save('data/%s/true_y_%.3fstep.npy' % (dataset, rstep), true_y)
                true_y0_test = true_y[-1]
                true_y = true_y[:-1]
                true_y_test = odeint(VL20(), true_y0_test, t[:steps_burn + steps_test + steps_valid],
                                     method='rk4', options={'step_size': step})
    # Two level Lorenz - just used the DAPPER implementation here since we didn't need the ability to
    # differentiably forward integrate
    elif dataset == 'lorenzuv':
        with torch.no_grad():
            if check_disk and os.path.exists('data/%s/true_y_%.3fstep.npy' % (dataset, rstep)):
                true_y = torch.Tensor(np.load('data/%s/true_y_%.3fstep.npy' % (dataset, rstep)))         
               
                true_y_test =true_y[-(steps_test + steps_valid):].unsqueeze(1)
                true_y = true_y[:-(steps_test + steps_valid)].unsqueeze(1)
            else:
               raise NotImplementedError
    # Not used in paper due to dimensionality making full rank EnKF too simple to compute
    # Also, there might be an error in this somewhere since it was never debugged
    elif dataset == 'lorenz63':
        with torch.no_grad():
            if check_disk and os.path.exists('data/%s/true_y_%.3fstep.npy' % (dataset, rstep)):
                true_y = torch.Tensor(np.load('data/%s/true_y_%.3fstep.npy' % (dataset, rstep)))         
                true_y0_test = true_y[-1]
                true_y = true_y[:-1]
                true_y_test = odeint(L63(), true_y0_test, t[:steps_valid + steps_test],
                                    method='rk4', options = {'step_size': step} )  
            else:
                true_y0 = torch.Tensor([[1.509, -1.531, 25.46]])
                true_y = odeint(L63(), true_y0, t, method='rk4', options = {'step_size': step} )
                if check_disk:
                    np.save('data/%s/true_y_%.3fstep.npy' % (dataset, rstep), true_y)
                true_y0_test = true_y[-1].unsqueeze(0)
                true_y = true_y[:-1]
                true_y_test = odeint(L63(), true_y0_test, t[:steps_test + steps_valid],
                            method='rk4', options = {'step_size': step} )   
    elif dataset == 'ks':
        with torch.no_grad():
            if check_disk and os.path.exists('data/%s/true_y_%.3fstep.npy' % (dataset, rstep)):
                true_y = torch.Tensor(np.load('data/%s/true_y_%.3fstep.npy' % (dataset, rstep)))         
                true_y0_test = true_y[-1].unsqueeze(0)
                # print('EEEEE', true_y0_test.shape)
                true_y = true_y[:-1].unsqueeze(1)
                true_y_test = custom_int(true_y0_test, etd_rk4_wrapper(), steps_test + steps_valid).unsqueeze(1)
            else:
                # Generate initial point from Dapper
                grid = 32 * np.pi * torch.linspace(0, 1, 128 + 1)[1:]
                x0_Kassam = torch.cos(grid / 16) * (1 + torch.sin(grid / 16))
                x0 = x0_Kassam.clone().unsqueeze(0)
                for _ in range(150):
                    x0 = custom_int(x0, etd_rk4_wrapper(), 150)[-1:]
                true_y0 = custom_int(x0, etd_rk4_wrapper(), 10**3)[-1:]
                true_y = custom_int(true_y0, etd_rk4_wrapper(), t.shape[0])
                if check_disk:
                    np.save('data/%s/true_y_%.3fstep.npy' % (dataset, rstep), true_y)
                true_y0_test = true_y[-1].unsqueeze(0)
                true_y = true_y[:-1].unsqueeze(1)
                true_y_test = custom_int(true_y0_test, etd_rk4_wrapper(), steps_test + steps_valid).unsqueeze(1)
    else:
        raise ValueError('Dataset not implemented')
    return true_y, true_y_test[steps_burn:steps_valid+steps_burn], true_y_test[steps_burn + steps_valid:]

def etd_rk4_wrapper(device = None, dt=0.5,DL=32,Nx=128):
    """ Returns an ETD-RK4 integrator for the KS equation. Currently very specific, need 
    to adjust this to fit into the same framework as the ODE integrators
    
    Directly ported from https://github.com/nansencenter/DAPPER/blob/master/dapper/mods/KS/core.py 
    which is adapted from kursiv.m of Kassam and Trefethen, 2002, doi.org/10.1137/S1064827502410633.
    """
    if device is None:
        device = torch.device('cpu')
    kk = np.append(np.arange(0,Nx/2),0)*2/DL                     # wave nums for rfft
    h = dt

    # Operators
    L = kk**2 - kk**4 # Linear operator for K-S eqn: F[ - u_xx - u_xxxx]

    # Precompute ETDRK4 scalar quantities
    E  = torch.Tensor(np.exp(h*L)).unsqueeze(0).to(device)    # Integrating factor, eval at dt
    E2 = torch.Tensor(np.exp(h*L/2)).unsqueeze(0).to(device)    # Integrating factor, eval at dt/2
    
    # Roots of unity are used to discretize a circular countour...
    nRoots = 16
    roots = np.exp( 1j * np.pi * (0.5+np.arange(nRoots))/nRoots ) 
    # ... the associated integral then reduces to the mean,
    # g(CL).mean(axis=-1) ~= g(L), whose computation is more stable.
    CL = h * L[:,None] + roots # Contour for (each element of) L
    # E * exact_integral of integrating factor:
    Q  = torch.Tensor(h * ( (np.exp(CL/2)-1)    / CL    ).mean(axis=-1).real).unsqueeze(0).to(device)
    # RK4 coefficients (modified by Cox-Matthews):
    f1 = torch.Tensor(h * ( (-4-CL+np.exp(CL)*(4-3*CL+CL**2)) / CL**3 ).mean(axis=-1).real).unsqueeze(0).to(device)
    f2 = torch.Tensor(h * (   (2+CL+np.exp(CL)*(-2+CL))       / CL**3 ).mean(axis=-1).real).unsqueeze(0).to(device)
    f3 = torch.Tensor(h * ( (-4-3*CL-CL**2+np.exp(CL)*(4-CL)) / CL**3 ).mean(axis=-1).real).unsqueeze(0).to(device)

    D = 1j*torch.Tensor(kk).to(device) # Differentiation to compute:  F[ u_x ]

    def NL(v, verb = False):
        return -.5 * D * torch.fft.rfft(torch.fft.irfft(v, dim=-1)**2, dim=-1)
        
    def inner(v, t, dt, verb = False):
        v = torch.fft.rfft(v, dim=-1)
        N1  = NL(v, verb)
        v1  = E2*v  + Q*N1

        N2a = NL(v1)
        v2a = E2*v  + Q*N2a
        
        N2b = NL(v2a)
        v2b = E2*v1 + Q*(2*N2b-N1)
        
        N3  = NL(v2b)
        v   = E*v  + N1*f1 + 2*(N2a+N2b)*f2 + N3*f3
        return torch.fft.irfft(v, dim=-1)
    return inner

def odeint_etd_wrapper(device = None, dt=0.5,DL=32,Nx=128):
    """ Kind of wasteful, but reduces code duplication elsewhere """
    ode_func = etd_rk4_wrapper(device, dt, DL, Nx)
    def inner(t, x0):
        x1 = ode_func(x0, dt, dt)
        x1 = ode_func(x1, dt, dt)
        return x1-x0
    return inner

# This basically is just a hack for KS training
def custom_int(x0, int_function, steps, dt = .5):
    out = [x0]
    x = x0
    for i in range(steps):
        x = int_function(x, None, dt)
        x = int_function(x, None, dt)
        out.append(x)
    return torch.cat(out, 0)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class TimeStack:
    def __call__(self, batch):
        return torch.cat(batch, dim = 1)

class ChunkedTimeseries(Dataset):
    """Chunked timeseries dataset."""

    def __init__(self, seq, chunk_size = 40, overlap = .25, transform=None):
        """
        Args:
            seq (torch.Tensor): Tensor containing time series
            chunk_size (int): size of chunks to produce
            overlap (float): 
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.seq = seq
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.n = seq.shape[0]
        self.starts = np.array([i * chunk_size for i in range(self.n//chunk_size)])
        self.transform = transform

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        start = min(self.starts[idx] + np.random.randint(int(self.overlap * self.chunk_size)), 
                        self.n - self.chunk_size)
        sample = self.seq[start:start+self.chunk_size]
        if self.transform:
            sample = self.transform(sample)
        return sample