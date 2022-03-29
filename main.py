from slinky import neuralnets as snn
from slinky import func as f
from slinky import misc as m
import argparse

from torchdiffeq import odeint_adjoint as odeint

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

parser = argparse.ArgumentParser('ODE slinky')
args = parser.parse_args()

# parameter settings
device = torch.device("cuda:1")

args.niters = int(1e3)
args.atol = 1e-4
args.rtol = 1e-4
args.save_data = True
args.save_model = True
args.save_freq = 10
args.test_freq = 10
args.num_cycles = 76
args.folder = 'data_slinky'
args.deltaT = 0.01
args.batch_time = 1+1
args.batch_size = 5
args.dataRecorder = './histRecorder.txt'

# read in the data and set the variables
true_y = m.read_data('./SlinkyGroundTruth',args.num_cycles)[::10,...].to(device)
args.data_size = true_y.size()[0] # the length of the entire data

true_y0 = true_y[0,:].to(device)
t = torch.linspace(0.,(true_y.size()[0]-1)*args.deltaT,true_y.size()[0]).to(device)

if args.save_data:
    m.makedirs(args.folder)

if __name__ == '__main__':

	m.seed_torch() # fix the random seed

	func_orig = f.ODEFunc(NeuronsPerLayer=32, NumLayer=5, Device=device).to(device) # the physical model
	func = f.ODEPhys(func_orig).to(device) # the class converting a second-order system into a first-order system, to be solved by ODE solvers

	optimizer = optim.Adam(func.parameters(), lr=1e-3)

	hist = [] # recording the loss history
	for itr in range(1, args.niters + 1):

		optimizer.zero_grad()

		batch_y0, batch_t, batch_y = m.get_batch(true_y,t,args.data_size,args.batch_time,args.batch_size)
		batch_y0, batch_t, batch_y = batch_y0.to(device), batch_t.to(device), batch_y.to(device)

		pred_y = odeint(func, batch_y0, batch_t, atol=args.atol, rtol=args.rtol).to(device)

		select_cycles = torch.from_numpy(np.random.choice(np.arange(args.num_cycles, dtype=np.int64), 20, replace=False))
		weight6 = torch.tensor([1e2,1e2,1,1e2,1e2,1]).to(device) ** 2
		loss = torch.mean(torch.abs(pred_y[...,select_cycles,0:] - batch_y[...,select_cycles,0:]) ** 2 * weight6)
        
		print('Iter {:04d} | Training Batch Loss {:.6f}'.format(itr, loss.item()))
		loss.backward()

		optimizer.step()

		# need to record the training loss at each timestep
		hist.append([args.batch_time,loss])
		np.savetxt(args.dataRecorder, np.array(hist))

		if True and itr % 2 == 0 and args.batch_time < 70:
			args.batch_time += 1
			print('increased the batch sequence length')
			print('The current lenght is {:04d}'.format(args.batch_time))

		if args.save_data and itr % args.test_freq == 0 and args.batch_time > 50:
			func.eval()
			pred_y = odeint(func, true_y0, t, atol=args.atol, rtol=args.rtol)
			loss = torch.mean(torch.abs(pred_y - true_y))
			print('Iter {:04d} | Training total Loss \033[1;32;43m{:.6f}\033[0m'.format(itr, loss.item()))
			m.save_data(true_y, pred_y, t, args.folder, itr, 'train')
			func.train()

		if args.save_model and itr % args.save_freq == 0 and args.batch_time > 10:
			func.eval()
			torch.save(func_orig.state_dict(), './slinky_func_orig.pt')
			func.train()

