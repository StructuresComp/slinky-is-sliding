import torch
import math

def removeRigidBodyMotion(data):
    BA1 = (data[...,2] + data[...,5])/2
    BA2 = (data[...,5] + data[...,8])/2
    
    col1 = torch.cos(BA1) * (data[...,3]-data[...,0]) + torch.sin(BA1) * (data[...,4]-data[...,1]) # xi_1
    col2 = -torch.sin(BA1) * (data[...,3]-data[...,0]) + torch.cos(BA1) * (data[...,4]-data[...,1]) # eta_1
    col3 = data[...,5] - data[...,2] # theta_1
    col4 = torch.cos(BA2) * (data[...,6]-data[...,3]) + torch.sin(BA2) * (data[...,7]-data[...,4]) # xi_2
    col5 = -torch.sin(BA2) * (data[...,6]-data[...,3]) + torch.cos(BA2) * (data[...,7]-data[...,4]) # eta_2
    col6 = data[...,8] - data[...,5] # theta_2
    return torch.stack((col1,col2,col3,col4,col5,col6),dim=-1)

def chiral_transformation_x(data):
    new_data = data.clone()
    new_data[...,0] = -data[...,0]
    new_data[...,2] = -data[...,2]
    new_data[...,3] = -data[...,3]
    new_data[...,5] = -data[...,5]
    return new_data

def chiral_transformation_y(data):
    new_data = data.clone()
    new_data[...,0] = data[...,3]
    new_data[...,1] = -data[...,4]
    new_data[...,2] = data[...,5]
    new_data[...,3] = data[...,0]
    new_data[...,4] = -data[...,1]
    new_data[...,5] = data[...,2]
    return new_data

def chiral_transformation_xy(data):
    new_data = data.clone()
    new_data = chiral_transformation_x(data)
    new_data = chiral_transformation_y(new_data)
    return new_data
