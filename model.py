import torch
import torch.nn as nn
from torch.nn import functional as F

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, z_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=128, kernel_size=7, padding=3, padding_mode='replicate')
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2, padding_mode='replicate')
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1, padding_mode='replicate')
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=out_dim, kernel_size=3, padding=1, padding_mode='replicate')
    
    def forward(self, input):

        b,t,c = input.size()

        x = F.relu((self.conv1(input.permute(0,2,1))))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x))) #b,128,t
        out = (self.conv4(x)).permute(0,2,1) #b,c,t
        
        return out