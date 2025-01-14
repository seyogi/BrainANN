import math
import time
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import pickle
import sys
import ANNModels.RNNnet
import ANNModels.MLPnet
import ANNModels.SynthesisMLP

def plot_all_ac(arr,hidden_size,color='#FF0000'):
    fig = plt.figure()
    for i in range(hidden_size):
        ax = fig.add_subplot(8, 8, i+1)
        tmp = arr[:,i]
        ax.plot(tmp[1:], color=color,alpha=1)
        ax.axis("off")

def plot_activity(i,ax,_pca,trial_infos,full_activity_small_dict):
    trial = trial_infos[i]
    activity_pc = _pca.transform(full_activity_small_dict[i][10:40])
    color = '#FFCCCC' if trial['v1'] <= 14 else '#FF9999' if trial['v1'] <= 20 else '#FF6666' if trial['v1'] < 30 else '#FF0000'
    _ = ax.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)

def plot_activity_FL(i,ax,_pca,trial_infos,full_activity_small_dict):
    activity_pc = _pca.transform(full_activity_small_dict[i][10:40])
    trial = trial_infos[i]
    ax.plot(activity_pc[0, 0], activity_pc[0, 1], '^', color="green", alpha=1)
    color = '#CCCCFF' if trial['v1'] <= 14 else '#9999FF' if trial['v1'] <= 20 else '#6666FF' if trial['v1'] < 30 else '#0000FF'
    ax.plot(activity_pc[-1, 0], activity_pc[-1, 1], 'x', color=color)
    
def plot_heatmap(cov, ax, title="tmp"):
    m, n = cov.shape
    im = ax.imshow(cov)
    ax.set_title(title)
    for i in range(m):
        for j in range(n):
            ax.text(j, i, f"{round(cov[i, j], 2)}", ha="center", va="center", color="w")
    plt.colorbar(im)
    
def main(mlp_model = None, time_length = 5):
    sys.modules['RNNnet'] = ANNModels.RNNnet
    sys.modules['MLPnet'] = ANNModels.MLPnet
    
    if mlp_model != None:
        mlp_net = mlp_model
    else:
        mlp_name = ".\ANNModels\MLP\MLP_" + "1210_1"
        file = open(mlp_name +'.pickle', mode='br')
        mlp_net1 = pickle.load(file)
        file.close()
        mlp_name = ".\ANNModels\MLP\MLP_" + "1115"
        file = open(mlp_name +'.pickle', mode='br')
        mlp_net = pickle.load(file)
        file.close()
        #mlp_net = ANNModels.SynthesisMLP.SynthesisMLP(mlp_net1,mlp_net2)
    
    activity_name = ".\ANNModels\RNNActivityDict\_tmp_" + "5_points"
    file = open(activity_name +'.pickle', mode='br')
    tmp_activity = pickle.load(file)
    file.close()

    new_activity_dict = tmp_activity[0]
    full_activity_small_dict = tmp_activity[1]
    trial_infos = tmp_activity[2] 

    hidden_size = 64
    pre_activity_dict = []

    num_trial = 200

    _tmp = []
    for _trial in full_activity_small_dict:
        _tmp.append(_trial[10:40])
    _activity = np.concatenate(list(_tmp[i] for i in range(num_trial)), axis=0)
    print('Shape of the neural activity: (Time points, Neurons): ', _activity.shape)
    _pca = PCA(n_components=2)
    _pca.fit(_activity)
    _pca = tmp_activity[3] 

    fig = plt.figure()
    axs = []
    for i in range(64):
        axs.append(fig.add_subplot(8, 8, i+1))
        axs[i].axis("off")
    
    maxnum = np.zeros(64)

    for train_index in range(50):
        pre_activity_dict = []
        trial = trial_infos[train_index]
        tmp = torch.from_numpy(np.vstack(new_activity_dict[(10-time_length)+train_index*(46-time_length)]["label"]).astype(np.double)).clone()
        pre_activity_dict.append(tmp[:,0].detach().numpy())

        for i in range(29):
            tmp = torch.from_numpy(np.vstack(new_activity_dict[(10-time_length)+1+i+train_index*(46-time_length)]["data"]).astype(np.double)).clone()
            #print(tmp.T)
            tmp = mlp_net(tmp.T).squeeze()
            pre_activity_dict.append(tmp.detach().numpy())
            

        for i in range(hidden_size):
            tmp = full_activity_small_dict[train_index][10:40][:,i]
            if maxnum[i] < np.max(tmp):
                maxnum[i] = np.max(tmp)
            tmp2 = np.array(pre_activity_dict)[:,i]
            color = '#FF0000' #true
            color = (
                "#FFCCCC"
                if trial["v1"] <= 14
                else (
                    "#FF9999"
                    if trial["v1"] <= 20
                    else "#FF6666" if trial["v1"] < 30 else "#FF0000"
                )
            )
            axs[i].plot(tmp[1:], color=color, alpha=1)
            color = (
                "#CCCCFF"
                if trial["v1"] <= 14
                else (
                    "#9999FF"
                    if trial["v1"] <= 20
                    else "#6666FF" if trial["v1"] < 30 else "#0000FF"
                )
            )
            axs[i].plot(tmp2[1:], color=color, alpha=1)
            
    #fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    #plot_heatmap(maxnum.reshape(8,8),ax,"10:40")
    """
    fig = plt.figure()
    for i in range(hidden_size):
        ax = fig.add_subplot(8, 8, i+1)
        tmp = full_activity_small_dict[train_index][10:40][:,i]
        tmp2 = np.array(pre_activity_dict)[:,i]
        color = '#0000FF' #pre
        ax.plot(tmp2[1:], color=color,alpha=1)
        ax.axis("off")

    fig = plt.figure()
    for i in range(hidden_size):
        ax = fig.add_subplot(8, 8, i+1)
        tmp = full_activity_small_dict[train_index][10:40][:,i]
        tmp2 = np.array(pre_activity_dict)[:,i]
        color = '#FF0000' #true
        ax.plot(tmp[1:], color=color,alpha=1)
        ax.axis("off") 
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(100):
        plot_activity(i,ax1,_pca,trial_infos,full_activity_small_dict)

    for i in range(100):
        plot_activity_FL(i,ax1,_pca,trial_infos,full_activity_small_dict)

    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')

    plot_activity(train_index,ax2,_pca,trial_infos,full_activity_small_dict)
    _pre_activity_dict = np.array(pre_activity_dict)
    _pre_activity_dict_pc = _pca.transform(_pre_activity_dict)

    print(len(_pre_activity_dict_pc[:, 0]))
    _ = ax1.plot(_pre_activity_dict_pc[:, 0], _pre_activity_dict_pc[:, 1], 'o-', color='#000000')
    _ = ax2.plot(_pre_activity_dict_pc[:, 0], _pre_activity_dict_pc[:, 1], 'o-', color='#000000')
    plot_activity_FL(train_index,ax2,_pca,trial_infos,full_activity_small_dict)

    plt.show()

if __name__ == "__main__":
    main()
