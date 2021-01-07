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
    """ Generates training and test data for given model 

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
                true_y = true_y[:-1].unsqueeze(1)
                true_y_test = custom_int(true_y0_test, etd_rk4_wrapper(), steps_test + steps_valid).unsqueeze(1)
            else:
                # Arbitrary point generated from random sampling and burn in. Used a consistent point to avoid
                # needing to burn in every time data was regenerated.
                true_y0 = torch.Tensor([[ 2.29686202e+00,  9.07480719e-01, -1.21535431e+00, -2.46597639e+00,
                                           -2.37382655e+00, -1.71890991e+00, -1.23758130e+00, -1.11598678e+00,
                                           -1.16453894e+00, -1.11813720e+00, -8.19511004e-01, -2.41798905e-01,
                                            5.43700297e-01,  1.30149669e+00,  1.53906452e+00,  7.73817686e-01,
                                           -7.21660730e-01, -1.87860814e+00, -2.02435873e+00, -1.44080672e+00,
                                           -7.16573648e-01, -2.20952734e-01, -5.96580541e-02, -1.57080157e-01,
                                           -3.15456291e-01, -3.12828107e-01, -1.00713424e-02,  5.91315511e-01,
                                            1.30884300e+00,  1.71446050e+00,  1.28252287e+00,  1.88342453e-08,
                                           -1.28252285e+00, -1.71446050e+00, -1.30884302e+00, -5.91315530e-01,
                                            1.00713291e-02,  3.12828101e-01,  3.15456292e-01,  1.57080162e-01,
                                            5.96580570e-02,  2.20952732e-01,  7.16573640e-01,  1.44080671e+00,
                                            2.02435873e+00,  1.87860815e+00,  7.21660756e-01, -7.73817668e-01,
                                           -1.53906452e+00, -1.30149670e+00, -5.43700316e-01,  2.41798889e-01,
                                            8.19510999e-01,  1.11813721e+00,  1.16453896e+00,  1.11598679e+00,
                                            1.23758128e+00,  1.71890986e+00,  2.37382650e+00,  2.46597644e+00,
                                            1.21535451e+00, -9.07480490e-01, -2.29686192e+00, -2.23904310e+00,
                                           -1.34378586e+00, -2.78424200e-01,  7.68161976e-01,  1.77975603e+00,
                                            2.42574991e+00,  1.98626067e+00,  2.28283747e-01, -1.68841340e+00,
                                           -2.42970158e+00, -2.03938025e+00, -1.30264199e+00, -6.80485452e-01,
                                           -1.63246561e-01,  4.09105753e-01,  1.01236550e+00,  1.24803367e+00,
                                            5.81167008e-01, -8.92285771e-01, -2.18463290e+00, -2.51042305e+00,
                                           -2.03375428e+00, -1.33100283e+00, -7.88227006e-01, -4.96947128e-01,
                                           -3.62235075e-01, -2.08197222e-01,  1.40131073e-01,  7.70184071e-01,
                                            1.56512259e+00,  2.05755972e+00,  1.56436914e+00, -9.64891121e-09,
                                           -1.56436915e+00, -2.05755971e+00, -1.56512258e+00, -7.70184056e-01,
                                           -1.40131062e-01,  2.08197229e-01,  3.62235079e-01,  4.96947128e-01,
                                            7.88227003e-01,  1.33100282e+00,  2.03375427e+00,  2.51042305e+00,
                                            2.18463290e+00,  8.92285782e-01, -5.81167006e-01, -1.24803369e+00,
                                           -1.01236553e+00, -4.09105782e-01,  1.63246542e-01,  6.80485460e-01,
                                            1.30264203e+00,  2.03938031e+00,  2.42970159e+00,  1.68841327e+00,
                                           -2.28283972e-01, -1.98626083e+00, -2.42574993e+00, -1.77975599e+00,
                                           -7.68161950e-01,  2.78424179e-01,  1.34378581e+00,  2.23904308e+00]])
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
    E  = torch.Tensor(np.exp(h*L)).unsqueeze(1).to(device)    # Integrating factor, eval at dt
    E2 = torch.Tensor(np.exp(h*L/2)).unsqueeze(1).to(device)    # Integrating factor, eval at dt/2
    
    # Roots of unity are used to discretize a circular countour...
    nRoots = 16
    roots = np.exp( 1j * np.pi * (0.5+np.arange(nRoots))/nRoots ) 
    # ... the associated integral then reduces to the mean,
    # g(CL).mean(axis=-1) ~= g(L), whose computation is more stable.
    CL = h * L[:,None] + roots # Contour for (each element of) L
    # E * exact_integral of integrating factor:
    Q  = torch.Tensor(h * (          (np.exp(CL/2)-1)         / CL    ).mean(axis=-1).real).unsqueeze(1).to(device)
    # RK4 coefficients (modified by Cox-Matthews):
    f1 = torch.Tensor(h * ( (-4-CL+np.exp(CL)*(4-3*CL+CL**2)) / CL**3 ).mean(axis=-1).real).unsqueeze(1).to(device)
    f2 = torch.Tensor(h * (   (2+CL+np.exp(CL)*(-2+CL))       / CL**3 ).mean(axis=-1).real).unsqueeze(1).to(device)
    f3 = torch.Tensor(h * ( (-4-3*CL-CL**2+np.exp(CL)*(4-CL)) / CL**3 ).mean(axis=-1).real).unsqueeze(1).to(device)

    D = torch.Tensor(kk).to(device) # Differentiation to compute:  F[ u_x ]
    
    def NL(v, verb = False):
        v_mult = torch.rfft(torch.irfft(v, 1, signal_sizes =( Nx, )) ** 2, 1 )
        vr, vi = torch.split(v_mult, 1, dim = -1)
        vr, vi = vr.squeeze(-1), vi.squeeze(-1)
#         if verb:
#             print(vr.device, D.device)
        vr *= D
        vi *= -D
        v = -.5 * torch.stack([vi, vr], -1)
        return v
        
    def inner(v, t, dt, verb = False):
        v = torch.rfft(v, 1)
        N1  = NL(v, verb)
#         print(E2.shape, v.shape, Q.shape, N1.shape)
        v1  = E2*v  + Q*N1
        
        N2a = NL(v1)
        v2a = E2*v  + Q*N2a
        
        N2b = NL(v2a) 
        v2b = E2*v1 + Q*(2*N2b-N1)
        
        N3  = NL(v2b)
        v   = E*v  + N1*f1 + 2*(N2a+N2b)*f2 + N3*f3
        return torch.irfft(v, 1, signal_sizes =( Nx, ))
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