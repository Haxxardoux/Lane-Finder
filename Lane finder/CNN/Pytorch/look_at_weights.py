import torch
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# my stuff
from dataloader import image_loader
from model_components import UNet

# boy, do i wish i didnt have to hardcode this. no idea why the model params files act so weird under for loops.

net_1 = UNet(3, 1, bilinear=True)
net_2 = UNet(3, 1, bilinear=True)
net_3 = UNet(3, 1, bilinear=True)
net_4 = UNet(3, 1, bilinear=True)
net_5 = UNet(3, 1, bilinear=True)
net_6 = UNet(3, 1, bilinear=True)
net_7 = UNet(3, 1, bilinear=True)

net_1.load_state_dict(torch.load('C://Users//turbo//Python projects//Lane-finder//Lane-Finder//CNN//checkpoints//CONST_NUM_FILCP_epoch1.pth'))
net_2.load_state_dict(torch.load('C://Users//turbo//Python projects//Lane-finder//Lane-Finder//CNN//checkpoints//CONST_NUM_FILCP_epoch2.pth'))
net_3.load_state_dict(torch.load('C://Users//turbo//Python projects//Lane-finder//Lane-Finder//CNN//checkpoints//CONST_NUM_FILCP_epoch3.pth'))
net_4.load_state_dict(torch.load('C://Users//turbo//Python projects//Lane-finder//Lane-Finder//CNN//checkpoints//CONST_NUM_FILCP_epoch4.pth'))
net_5.load_state_dict(torch.load('C://Users//turbo//Python projects//Lane-finder//Lane-Finder//CNN//checkpoints//CONST_NUM_FILCP_epoch5.pth'))
net_6.load_state_dict(torch.load('C://Users//turbo//Python projects//Lane-finder//Lane-Finder//CNN//checkpoints//CONST_NUM_FILCP_epoch6.pth'))
net_7.load_state_dict(torch.load('C://Users//turbo//Python projects//Lane-finder//Lane-Finder//CNN//checkpoints//CONST_NUM_FILCP_epoch7.pth'))


net_1_params = [] 
for params in net_1.parameters():
    net_1_params.append(params.detach().numpy())
net_1_params = np.array(net_1_params[::4])

net_2_params = [] 
for params in net_2.parameters():
    net_2_params.append(params.detach().numpy())
net_2_params = np.array(net_2_params[::4])

net_3_params = [] 
for params in net_3.parameters():
    net_3_params.append(params.detach().numpy())
net_3_params = np.array(net_3_params[::4])

net_4_params = [] 
for params in net_4.parameters():
    net_4_params.append(params.detach().numpy())
net_4_params = np.array(net_4_params[::4])

net_5_params = [] 
for params in net_5.parameters():
    net_5_params.append(params.detach().numpy())
net_5_params = np.array(net_5_params[::4])

net_6_params = [] 
for params in net_6.parameters():
    net_6_params.append(params.detach().numpy())
net_6_params = np.array(net_6_params[::4])

net_7_params = [] 
for params in net_7.parameters():
    net_7_params.append(params.detach().numpy())
net_7_params = np.array(net_7_params[::4])

net_list = []
for i in range(18):
    layer_list = []
    net_2_diff = np.mean(net_2_params[i] - net_1_params[i])
    layer_list.append(net_2_diff)
    net_3_diff = np.mean(net_3_params[i] - net_2_params[i])
    layer_list.append(net_3_diff)
    net_4_diff = np.mean(net_4_params[i] - net_3_params[i])
    layer_list.append(net_4_diff)
    net_5_diff = np.mean(net_5_params[i] - net_4_params[i])
    layer_list.append(net_5_diff)
    net_6_diff = np.mean(net_6_params[i] - net_5_params[i])
    layer_list.append(net_6_diff)
    net_7_diff = np.mean(net_7_params[i] - net_6_params[i])
    layer_list.append(net_7_diff)
    net_list.append(layer_list)

net_array = np.array(net_list)

from seaborn import heatmap
from pandas import DataFrame

means_df = DataFrame(net_array)
heatmap(means_df, cmap="Blues")
plt.show()
