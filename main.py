import torch
from torch import nn, optim
from torch.nn import functional as F
import argparse
import datetime
import os
from tqdm import tqdm 
import numpy as np
from matplotlib import pyplot as plt
import wandb
import torchvision.utils as vutils
from model import AutoEncoder
from physics import compute_a, compute_w
from utils import sample_noise, DataBatch
        
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z', type=int, default=10, help="Number of latent dimensions")
parser.add_argument('--pre_epochs', type=int, default=100, help="Number of pre-training epochs")
parser.add_argument('--epochs', type=int, default=10000, help="Number of training epochs")
parser.add_argument('--model_path', default='./checkpoint/', help='Directory for saving model')
parser.add_argument('--run_tag', default='PILOT_IMU', help='Tag for the current run')
parser.add_argument('--rate', type=float, default=0.01, help="Manually added noise ratio")
parser.add_argument('--batchSize', type=int, default=16, help="Batch size")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset
high_vi = torch.from_numpy(np.load('./data/high_vi.npy')).float().to(device)
high_w = torch.from_numpy(np.load('./data/high_w.npy')).float().to(device)
high_a = torch.from_numpy(np.load('./data/high_a.npy')).float().to(device)

low_vi = torch.from_numpy(np.load('./data/low_vi.npy')).float().to(device)
low_w = torch.from_numpy(np.load('./data/low_w.npy')).float().to(device)
low_a = torch.from_numpy(np.load('./data/low_a.npy')).float().to(device)

seq_len, vi_dim, imu_w_dim, imu_a_dim = high_vi.shape[1], high_vi.shape[2], high_w.shape[2], high_a.shape[2]

# define model
model = AutoEncoder(in_dim=vi_dim+imu_w_dim+imu_a_dim, z_dim=args.z, out_dim=vi_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
mse = nn.MSELoss().to(device)
mae = nn.L1Loss().to(device)

# initial difference
vi_w = compute_w(high_vi)
vi_a = compute_a(high_vi)

init_w_loss = mse(vi_w, high_w).item()
init_a_loss = mse(vi_a, high_a).item()
init_w_mae = mae(vi_w, high_w).item()
init_a_mae = mae(vi_a, high_a).item()

print('initial w_loss: %.4f a_loss: %.4f w_mae: %.4f a_mae: %.4f' % (init_w_loss, init_a_loss, init_w_mae, init_a_mae))

vi_w = compute_w(low_vi)
vi_a = compute_a(low_vi)

init_w_loss = mse(vi_w, low_w).item()
init_a_loss = mse(vi_a, low_a).item()
init_w_mae = mae(vi_w, low_w).item()
init_a_mae = mae(vi_a, low_a).item()

print('initial w_loss: %.4f a_loss: %.4f w_mae: %.4f a_mae: %.4f' % (init_w_loss, init_a_loss, init_w_mae, init_a_mae))

# logging
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
wandb.init(
    project='Controllable-Generation',
    config=vars(args),
    name=f"{args.run_tag}_{date}" 
)

# pre-training, phase I
print('################################# Phase I #################################')
best_loss = None
for epoch in range(args.pre_epochs):

    total_loss, total_vi_loss, total_w_loss, total_a_loss = 0, 0, 0, 0
    for batch_vi, batch_w, batch_a in DataBatch(high_vi, high_w, high_a, args.batchSize):
        
        noise_vi, idx = sample_noise(batch_vi, args.rate, device)
        input = torch.cat((noise_vi, batch_w, batch_a), dim=-1)
        denoise_vi = model(input)

        loss = mse(denoise_vi[:,idx], batch_vi[:,idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += len(batch_vi) * loss.item()
    
    total_loss /= len(high_vi)

    print('[%d/%d] pretrain total_loss: %.4f' % (epoch, args.pre_epochs, total_loss))
    wandb.log({"pretrain_loss": total_loss})
    
    if best_loss is None or total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), args.model_path + args.run_tag + 'pretrain_model')

model.load_state_dict(torch.load(args.model_path + args.run_tag + 'pretrain_model'))

# training, phase II
print('################################# Phase II #################################')
for epoch in range(args.epochs):

    total_loss, total_vi_loss, total_w_loss, total_a_loss = 0, 0, 0, 0
    for batch_vi, batch_w, batch_a in DataBatch(high_vi, high_w, high_a, args.batchSize):

        noise_vi, idx = sample_noise(batch_vi, args.rate, device)
        input = torch.cat((noise_vi, batch_w, batch_a), dim=-1)
        denoise_vi = model(input)
     
        vi_w = compute_w(denoise_vi)
        vi_a = compute_a(denoise_vi)

        vi_loss = mse(denoise_vi[:,idx], batch_vi[:,idx]) 
        w_loss = mse(vi_w, batch_w)
        a_loss = mse(vi_a, batch_a)
        imu_loss = w_loss + a_loss

        w1 = float(vi_loss.item()/w_loss.item())
        w2 = float(vi_loss.item()/a_loss.item())
     
        loss = vi_loss + w1*w_loss + w2*a_loss 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += len(batch_vi) * loss.item()
        total_vi_loss += len(batch_vi) * vi_loss.item()
        total_w_loss += len(batch_vi) * w_loss.item()
        total_a_loss += len(batch_vi) * a_loss.item()
    
    total_loss /= len(high_vi)
    total_vi_loss /= len(high_vi)
    total_w_loss /= len(high_vi)
    total_a_loss /= len(high_vi)

    print('[%d/%d] train total_loss: %.4f vi_loss: %.4f w_loss: %.4f a_loss: %.4f' % \
        (epoch, args.epochs, total_loss, total_vi_loss, total_w_loss, total_a_loss))
    wandb.log({"vi_train_loss": total_vi_loss, "w_train_loss": total_w_loss, "a_train_loss": total_a_loss})

    torch.save(model.state_dict(), args.model_path + args.run_tag + 'model')

    # test
    total_loss, total_vi_loss, total_w_loss, total_a_loss = 0, 0, 0, 0
    denoise_vi_list, vi_w_list, vi_a_list = [], [], []
    for batch_vi, batch_w, batch_a in DataBatch(low_vi, low_w, low_a, args.batchSize, shuffle=False):

        input = torch.cat((batch_vi, batch_w, batch_a), dim=-1)
        denoise_vi = model(input)

        vi_w = compute_w(denoise_vi)
        vi_a = compute_a(denoise_vi)

        denoise_vi_list.append(denoise_vi)
        vi_w_list.append(vi_w)
        vi_a_list.append(vi_a)

        vi_loss = mse(denoise_vi, batch_vi)
        w_loss = mse(vi_w, batch_w)
        a_loss = mse(vi_a, batch_a)
        
        loss = w_loss + a_loss

        total_loss += len(batch_vi) * loss.item()
        total_vi_loss += len(batch_vi) * vi_loss.item()

    total_loss /= len(low_vi)
    total_vi_loss /= len(low_vi)

    denoise_vi = torch.cat(denoise_vi_list)
    vi_w = torch.cat(vi_w_list)
    vi_a = torch.cat(vi_a_list)

    total_w_loss = mse(vi_w, low_w).item()
    total_a_loss = mse(vi_a, low_a).item()

    total_w_mae = mae(vi_w, low_w).item()
    total_a_mae = mae(vi_a, low_a).item()

    print('test total_loss: %.4f vi_loss: %.4f w_loss: %.4f a_loss: %.4f w_mae: %.4f a_mae: %.4f' % \
        (total_loss, total_vi_loss, total_w_loss, total_a_loss, total_w_mae, total_a_mae))
    wandb.log({"vi_loss": total_vi_loss, "w_loss": total_w_loss, "a_loss": total_a_loss, "w_mae": total_w_mae, "a_mae": total_a_mae})