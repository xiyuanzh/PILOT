import numpy as np
from scipy.signal import medfilt2d
from copy import deepcopy
import torch

def DataBatch(c1, c2, c3, batchsize, shuffle=True):
    
    n = c1.shape[0]
    if shuffle:
        index = np.random.permutation(n)
    else:
        index = np.arange(n)
    for i in range(int(np.ceil(n/batchsize))):
        inds = index[i*batchsize : min(n,(i+1)*batchsize)]
        yield c1[inds], c2[inds], c3[inds]

def multi_series_to_plot(vi, imu_w, imu_a, vi_recons, vi_w, vi_a, folder, title):
   
    fig = plt.figure(figsize=[50,8])
    ax0 = fig.add_subplot(241)
    ax0.plot(vi[:,0],label='vi_trans_x') 
    ax0.plot(vi[:,1],label='vi_trans_y') 
    ax0.plot(vi[:,2],label='vi_trans_z') 
    ax0.plot(vi_recons[:,0],label='vi_recons_trans_x')
    ax0.plot(vi_recons[:,1],label='vi_recons_trans_y')
    ax0.plot(vi_recons[:,2],label='vi_recons_trans_z')
    ax0.legend()

    ax1 = fig.add_subplot(242)
    ax1.plot(vi[:,3],label='vi_rot_x') 
    ax1.plot(vi[:,4],label='vi_rot_y') 
    ax1.plot(vi[:,5],label='vi_rot_z') 
    ax1.plot(vi[:,6],label='vi_rot_w') 
    ax1.plot(vi_recons[:,3],label='vi_recons_rot_x')
    ax1.plot(vi_recons[:,4],label='vi_recons_rot_y')
    ax1.plot(vi_recons[:,5],label='vi_recons_rot_z')
    ax1.plot(vi_recons[:,6],label='vi_recons_rot_w')
    ax1.legend()
  
    ax2 = fig.add_subplot(243)
    #vi_w_smooth = medfilt2d(vi_w[:,0][:,np.newaxis], [15,1])[:,0]
    ax2.plot(imu_w[:,0],color='c',label='imu_w_x') 
    ax2.plot(vi_w[:,0],color='y',label='vi_w_x')
    ax2.legend()

    ax3 = fig.add_subplot(244)
    #vi_a_smooth = medfilt2d(vi_a[:,0][:,np.newaxis], [15,1])[:,0]
    ax3.plot(imu_a[:,0],color='c',label='imu_a_x') 
    ax3.plot(vi_a[:,0],color='y',label='vi_a_x')
    ax3.legend()

    ax4 = fig.add_subplot(245)
    #vi_w_smooth = medfilt2d(vi_w[:,1][:,np.newaxis], [15,1])[:,0]
    ax4.plot(imu_w[:,1],color='c',label='imu_w_y') 
    ax4.plot(vi_w[:,1],color='y',label='vi_w_y')
    ax4.legend()

    ax5 = fig.add_subplot(246)
    #vi_a_smooth = medfilt2d(vi_a[:,1][:,np.newaxis], [15,1])[:,0]
    ax5.plot(imu_a[:,1],color='c',label='imu_a_y') 
    ax5.plot(vi_a[:,1],color='y',label='vi_a_y')
    ax5.legend()

    ax6 = fig.add_subplot(247)
    #vi_w_smooth = medfilt2d(vi_w[:,2][:,np.newaxis], [15,1])[:,0]
    ax6.plot(imu_w[:,2],color='c',label='imu_w_z') 
    ax6.plot(vi_w[:,2],color='y',label='vi_w_z')
    ax6.legend()

    ax7 = fig.add_subplot(248)
    #vi_a_smooth = medfilt2d(vi_a[:,2][:,np.newaxis], [15,1])[:,0]
    ax7.plot(imu_a[:,2],color='c',label='imu_a_z') 
    ax7.plot(vi_a[:,2],color='y',label='vi_a_z')
    ax7.legend()

    plt.savefig('./%s/%s.pdf'%(folder,title))

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def count_memory(model):
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    return mem, mem_params, mem_bufs

def sample_noise(data, rate, device):
    
    noise = 0.1 * torch.randn((data.shape[0], data.shape[1], data.shape[2])).to(device)
   
    idx1 = np.random.choice(data.shape[1], int(rate*data.shape[1]))
    noise_data = deepcopy(data)
    noise_data[:,idx1] = data[:,idx1] + noise[:,idx1]

    idx2 = np.random.choice(range(1,data.shape[1]), int(rate*(data.shape[1]-1)))
    noise_data[:,idx2] = noise_data[:,idx2-1]

    return noise_data, np.concatenate((idx1,idx2))

def z_norm(data):
    mean, var = data.mean(dim=(0,1)).unsqueeze(0).unsqueeze(1), data.std(dim=(0,1)).unsqueeze(0).unsqueeze(1)
    data = (data - mean) / (var + 1e-6)
    return data, mean, var

def de_norm(data, mean, var):
    return var * data + mean