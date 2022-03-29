import torch
import torch.nn as nn
import numpy as np
from slinky import neuralnets as snn
from slinky import transformation as trans

class ODEFunc(nn.Module):

	def __init__(self, NeuronsPerLayer=32, NumLayer=5, Boundaries=torch.tensor([1,1]), Device=torch.device("cpu")): # for boundary condition, 1 stands for fixed, 0 stands for free
		super(ODEFunc, self).__init__()
		self.neuronsPerLayer = NeuronsPerLayer
		self.numLayers = NumLayer
		self.net = nn.Sequential(
			nn.Linear(6,self.neuronsPerLayer),
			snn.DenseBlock(self.neuronsPerLayer,self.numLayers), 
			snn.Square(),
			nn.Linear(int(self.neuronsPerLayer*(self.numLayers+1)),1)
		)

		self.boundaries = Boundaries
		self.device = Device

		# initializations
		for m in self.net.modules():
			if isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, mean=0, std=0.01)
				nn.init.constant_(m.bias, val=0.01)

	def construct_y(self, y):
		if self.boundaries[0] == 1 and self.boundaries[1] == 1: # both ends fixed
			yp = y[...,1:-1,0:3].clone().requires_grad_(True)
			ypp = y[...,0:-2,0:3].clone()
			ypa = y[...,2:,0:3].clone()
		elif self.boundaries[0] == 0 and self.boundaries[1] == 1: # left end free, right end fixed
			yp = y[...,0:-1,0:3].clone().requires_grad_(True)
			ypp = y[...,0:-1,0:3].clone()
			ypp = torch.cat((ypp[...,0:1,:],ypp),-2)
			ypa = y[...,1:,0:3].clone()
		elif self.boundaries[0] == 1 and self.boundaries[1] == 0: # right end free, left end fixed
			yp = y[...,1:,0:3].clone().requires_grad_(True)
			ypp = y[...,0:-1,0:3].clone()
			ypa = y[...,2:,0:3].clone()
			ypa = torch.cat((ypa,ypa[...,-1:,:]),-2)
		elif self.boundaries[0] == 0 and self.boundaries[1] == 0: # both ends free
			yp = y[...,:,0:3].clone().requires_grad_(True)
			ypp = y[...,0:-1,0:3].clone()
			ypp = torch.cat((ypp[...,0:1,:],ypp),-2)
			ypa = y[...,1:,0:3].clone()
			ypa = torch.cat((ypa,ypa[...,-1:,:]),-2)
		else:
			raise RuntimeError("boundary conditions not allowed")
			yp = ypp = ypa = y.clone()
		return (yp, ypp, ypa)

	def contruct_grad(self, grad, aug):
		if self.boundaries[0] == 1 and self.boundaries[1] == 1: # both ends fixed
			return torch.cat((aug,grad,aug),-2)
		elif self.boundaries[0] == 0 and self.boundaries[1] == 1: # left end free, right end fixed
			return torch.cat((grad,aug),-2)
		elif self.boundaries[0] == 1 and self.boundaries[1] == 0: # right end free, left end fixed
			return torch.cat((aug,grad),-2)
		elif self.boundaries[0] == 0 and self.boundaries[1] == 0: # both ends free
			return grad
		else:
			raise RuntimeError("boundary conditions not allowed")
			return grad
    	
	def preprocessing(self, y):
		x = trans.removeRigidBodyMotion(y) # rigid body motion removal module
		augmented_x = torch.stack([x, trans.chiral_transformation_x(x), trans.chiral_transformation_y(x), trans.chiral_transformation_xy(x)], dim=0) # chirality module
		return augmented_x

	def calDeriv(self, y):
		with torch.enable_grad():
			# calculating the acceleration of the 2D slinky system
			# the dimensions of y are (num_samples, num_cycles, [x_dis,y_dis,alpha_dis,x_vel,y_vel,alpha_vel])
			yp, ypp, ypa = self.construct_y(y)
			yinput = torch.cat((ypp,yp,ypa),-1)
			augmented_x = self.preprocessing(yinput)
			out = self.net(augmented_x * torch.tensor([1e3,1e3,1e1,1e3,1e3,1e1]).to(self.device)) # adding weights here because the amplitude of deltaX and deltaY is ~0.01 (m), the amplitude of angle is ~pi
			out = torch.sum(out, dim=0, keepdim=False)
			deriv = torch.autograd.grad([out.sum()],[yp],retain_graph=True,create_graph=True) # "backward in forward" feature, deriving the equivariant force from the invariant energy
			# the dimensions of grad are (..., num_samples, num_cycles, 3)
			grad = deriv[0]
			if grad is not None:
				aug = torch.zeros_like(grad)[...,0:1,:]
				return self.contruct_grad(grad, aug)

	def forward(self, y):
		grad = self.calDeriv(y)
		if grad is not None:
			# the dimensions of the return value are (num_samples, num_cycles, 6)
			return grad

class ODEPhys(nn.Module):

	def __init__(self, ODEFunc, Boundaries=torch.tensor([1,1]), Device=torch.device("cpu")):
		super(ODEPhys, self).__init__()
		self.device = Device
		# the physical parameters 
		self.m = 2.5e-3
		self.J = 0.5*self.m*0.033**2
		self.g = 9.8
		# register constant matrices
		self.register_buffer('coeffMatrix1',torch.zeros(6,6).float())
		self.coeffMatrix1[3:,:3] = torch.eye(3).float()
		self.register_buffer('coeffMatrix2',torch.zeros(6,6).float())
		self.coeffMatrix2[0:3,3:] = -torch.diag(torch.tensor([1/self.m,1/self.m,1/self.J])).float()
		self.register_buffer('gVec',torch.tensor([0,0,0,0,-self.g,0]))
		self.ODEFunc = ODEFunc
		self.boundaries = Boundaries

	def construct_g(self, y):
		gravityGrad = torch.zeros_like(y)
		if self.boundaries[0] == 1 and self.boundaries[1] == 1: # both ends fixed
			gravityGrad[...,1:-1,:] = self.gVec
		elif self.boundaries[0] == 0 and self.boundaries[1] == 1: # left end free, right end fixed
			gravityGrad[...,0:-1,:] = self.gVec
		elif self.boundaries[0] == 1 and self.boundaries[1] == 0: # right end free, left end fixed
			gravityGrad[...,1:,:] = self.gVec
		elif self.boundaries[0] == 0 and self.boundaries[1] == 0: # both ends free
			gravityGrad[...,:,:] = self.gVec
		else:
			error("boundary conditions not allowed")
		return gravityGrad

	def forward(self, t, y):
		grad = self.ODEFunc(y)
		gravityGrad = self.construct_g(y)
		if grad is not None:
			# the dimensions of the return value are (num_samples, num_cycles, 6)
			acc = torch.matmul(y,self.coeffMatrix1) + torch.matmul(torch.cat((grad,torch.zeros_like(grad)),-1),self.coeffMatrix2) + gravityGrad
			return acc
