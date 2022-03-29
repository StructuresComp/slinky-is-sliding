import random
import os
import torch
import numpy as np

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_model(model, name="slinky"):
    traced_script_module = torch.jit.script(model.to("cpu"))
    traced_script_module = traced_script_module.to("cpu")
    traced_script_module.save("./"+name+".pt")

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def read_data(folder='SlinkyGroundTruth', num_cycles=80):
    true_y = np.loadtxt('./'+folder+'/helixCoordinate_2D.txt',delimiter=',')
    true_v = np.loadtxt('./'+folder+'/helixVelocity_2D.txt',delimiter=',')
    true_y = torch.from_numpy(true_y).float()
    true_v = torch.from_numpy(true_v).float()
    true_y = torch.reshape(true_y,(true_y.size()[0],num_cycles,3))
    true_v = torch.reshape(true_v,(true_v.size()[0],num_cycles,3))
    return torch.cat((true_y, true_v),-1)

def save_data(true_y, pred_y, t, folder, itr, append_name):
    f_true_y = './'+ folder + '/true_'+str(itr)+'_'+append_name+'.txt'
    f_pred_y = './'+ folder + '/pred_'+str(itr)+'_'+append_name+'.txt'
    f_t = './'+ folder + '/t_'+str(itr)+'_'+append_name+'.txt'
    true_y_record = true_y.detach().cpu().numpy().transpose(1,2,0)
    pred_y_record = pred_y.detach().cpu().numpy().transpose(1,2,0)

    np.savetxt(f_true_y, true_y_record.reshape((-1,true_y_record.shape[2])).transpose(), delimiter=",")
    np.savetxt(f_pred_y, pred_y_record.reshape((-1,pred_y_record.shape[2])).transpose(), delimiter=",")
    np.savetxt(f_t, np.squeeze(t.detach().cpu().numpy()), delimiter=",")

def get_batch(true_data,t,data_size,batch_time,batch_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False)) # M = num_batch_size
    batch_y0 = true_data[s]  # (M, D=[num_Cycles,6])
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_data[s + i,...] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y
    