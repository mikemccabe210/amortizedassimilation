import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import time
import numpy as np
import obs_configs

from torchdiffeq import odeint
from torch.utils.data import DataLoader
from amortized_assimilation.data_utils import ChunkedTimeseries, L96, L63, TimeStack, gen_data
from amortized_assimilation.models import MultiObs_KSConvEnAF
from amortized_assimilation.operators import filter_obs, mystery_operator

def train(epoch, loader, noise, m, model, optimizer, scheduler, obs_dict, device):
    """ Training loop """
    ntypes = len(obs_dict)
    for batch_y0 in loader:
        batch_y0 = batch_y0.to(device = device)
        optimizer.zero_grad()
        # Save noiseless data for comparison
        noiseless = batch_y0.clone().detach()
        # Generate noisy batch
        batch_y0 += torch.randn_like(batch_y0) * noise 
        states = torch.zeros_like(batch_y0)[0]
        
        # Sample from prior
        pred_y1 = noiseless[0].unsqueeze(1).repeat(1, m, 1)
        pred_y1 = pred_y1 + torch.randn_like(pred_y1)  * noise
#         print(pred_y1.shape)
        # Store everything
        preds_y, preds_y1, filts_y, filts_y1 = [], [], [], []
        preds_y_filt, preds_y1_filt = [], []
        known_inds_t = []
        known_inds_tp1 = []
        
        # Init components
        memory = torch.zeros(pred_y1.shape[0] * m, 4, 128, device = device)
        next_type = np.random.randint(0, ntypes)
        # Iterate over timesteps in batch
        for i, x in enumerate(batch_y0):
            # Check if observation operator is known
            if known_h(next_type) and i > 0:
                known_inds_t.append(i)
                known_inds_tp1.append(i-1)
            # Gen observations
            i_type = next_type 
            x = obs_dict[str(i_type % ntypes)](x).unsqueeze(1).repeat(1, m, 1)
            y = obs_dict[str(i_type % ntypes)](batch_y0[i])
            # Execute model
            # Due to KS hackyness, some of this is hardcoded
            pred_y, pred_y1, ens, memory = model(x, pred_y1, memory, 
                                      obs_type = str(i_type % ntypes))
        
            # Build outputs
            preds_y += [pred_y]
            filts_y += [y]
            preds_y_filt += [obs_dict[str(i_type % ntypes)](pred_y)]
            next_type = np.random.randint(0, ntypes)
            preds_y1_filt += [obs_dict[str((next_type) % ntypes)](pred_y1)]
            preds_y1 += [pred_y1]
            
        # Concat outputs
        # pred_y_list = torch.stack(preds_y)
        # pred_y1_list = torch.stack(preds_y1)
        filtered_pred = torch.stack(preds_y_filt)
        filtered_pred_y1 = torch.stack(preds_y1_filt)
        filt_y = torch.stack(filts_y)

        # Loss functions
        # noisy_analysis_loss = torch.mean(torch.sum((filtered_pred[1:] - filt_y[1:])**2, dim = 2))
        forecast_loss = torch.mean(torch.sum((filtered_pred_y1[known_inds_tp1].mean(dim = 2) 
                                      - filt_y[known_inds_t])**2, dim = 2))
        forecast_loss.backward()

        optimizer.step()
        scheduler.step()


def test(epoch, start_time, base_data, noise, m, model, obs_dict, device):
    """Test loop"""
    with torch.no_grad():
        state = base_data[0] 
        state = state.unsqueeze(1).repeat(1, m, 1) 
        state += torch.randn_like(state) * noise
        state = state.to(device = device)
        noisy_test = base_data + torch.randn_like(base_data) * noise
        noisy_test = noisy_test.to(device = device)
        pred_y_test, _, _, ens = assimilate_unseen_obs_ens(model, noisy_test, state, m,
                                                           obs_dict, device)
        loss = torch.mean(torch.mean((pred_y_test.cpu() - base_data.squeeze())**2, dim = 1)**.5)
        print('Iter {:04d} | Total Loss {:.6f} | Time {:.1f}'.format(epoch, loss.item(), time.time() - start_time))
        return loss.detach()
    
def assimilate_unseen_obs_ens(model, data, state, m, obs_dict, device):
    """ Executes online assimilation"""
    preds = []
    states = []
    # filtered_preds = []
    # filtered_obs = []
    ensembles = []
    memory = torch.zeros(m, 4, 128, device = device)
    
    for i, obs in enumerate(data):
        obs = obs_dict[str(i % len(obs_dict))](obs).unsqueeze(1).repeat(1, m, 1)
        pred, state, ens, memory = model(obs, state, memory, 
                                               obs_type = str(i % len(obs_dict)), 
                                              interval = torch.Tensor([0., .12]))
        states.append(state)
        preds.append(pred)
        ensembles.append(ens)
        i += 1
    return (torch.stack(preds, dim = 0).squeeze(), torch.stack(states, dim = 0).squeeze(),
            memory, torch.stack(ensembles, dim = 0).squeeze())
    
    
if __name__ == '__main__':
    
    class dummy():
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', type=int, default=240_000)
    parser.add_argument('--step_size', type=float, default=.5)
    parser.add_argument('--batch_steps', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--m', type=int, default=8)
    parser.add_argument('--n', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=24)
    parser.add_argument('--noise', type=float, default=1.)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--steps_valid', type=int, default=1000)
    parser.add_argument('--steps_test', type=int, default=5000)
    parser.add_argument('--check_disk', action='store_true')
    parser.add_argument('--obs_conf', type=str, default = 'full_obs')
    parser.add_argument('--do', type=float, default = .2)
    parser.add_argument('--device', type=str, default = 'gpu')
    args = parser.parse_args()

    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    t = torch.arange(0, args.train_steps*args.step_size, args.step_size)
    true_y, true_y_valid, true_y_test = gen_data('ks', t, args.steps_test, args.steps_valid, step = .5,
                                                  check_disk = True)

    print(true_y.shape)
    input_types, obs_dict, known_h = obs_configs.KS_configs[args.obs_conf]
    ntypes = len(obs_dict)
    # Set up model - ODE is hardcoded so parameterized dynamics here doesn't matter
    model = MultiObs_KSConvEnAF(args.n,args. hidden_size, ode_func = L63(),
                         input_types=input_types, m = args.m, device = device, do = args.do)
    model = model.to(device = device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay = 0)
    dummy_sched = dummy()
    dummy_sched.step = lambda: None
    data = ChunkedTimeseries(true_y, args.batch_steps, .95)
    loader = DataLoader(data, batch_size=args.batch_size,
                        shuffle=True, num_workers=0, collate_fn = TimeStack())
    
    start_time = time.time()
    # Training
    for itr in range(1, args.epochs + 1):
        if itr <= 50:
#             pass
            optimizer.param_groups[0]['lr'] *= 1.03
        else:
            optimizer.param_groups[0]['lr'] *= .993
        train(itr, loader, args.noise, args.m, model, optimizer, dummy_sched, obs_dict, device)
        test(itr, start_time, true_y_valid, args.noise, args.m, model, obs_dict, device)
        start_time = time.time()
    print('---Test set Results---')
    test(itr, start_time, true_y_test, args.noise, args.m, model, obs_dict, device)
        
        