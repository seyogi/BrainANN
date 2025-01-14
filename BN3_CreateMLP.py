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
from torch.optim import lr_scheduler
import ANNModels.MLPnet as MLPnet
import pickle

## Trainデータの作成
def mk_traindata(activity_name, time_length = 5):
    file = open(activity_name +'.pickle', mode='br')
    tmp_activity = pickle.load(file)
    file.close()

    new_activity_dict = tmp_activity[0]
    full_activity_small_dict = tmp_activity[1]
    trial_infos = tmp_activity[2] 
    print(len(tmp_activity[0]))
    # 学習時に見る time step
    num_trial = 2000

    tmp_list = [j for j in range(num_trial*(46-time_length))]
    random.shuffle(tmp_list)
    train_data = []
    for i in tmp_list:
        train_data.append([new_activity_dict[i]["data"], new_activity_dict[i]["label"]])
        
    tmp = np.array(train_data)
    data = tmp[:,0]
    label = tmp[:,1]
    data = np.vstack(data).astype(np.double)
    label = np.vstack(label).astype(np.double)
    print(len(data))
    return data,label

## MLPの訓練
def train_mlp(mlp_net,data,label):
    # Use Adam optimizer
    optimizer = optim.Adam(mlp_net.parameters(), lr=0.001)
    #step_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    criterion = nn.MSELoss()
    # Train for only one epoch
    running_loss = 0
    batch_size = 64
    num_trial = 1200
    for i in range(num_trial):
        tmp_activity = torch.from_numpy(data[i*batch_size : i*batch_size + batch_size]).clone()
        tmp_activity_next = torch.from_numpy(label[i*batch_size : i*batch_size + batch_size]).clone()
        output = mlp_net(tmp_activity)
        loss = criterion(output, tmp_activity_next)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #step_scheduler.step()
        running_loss += loss.item()
        
        printtime = num_trial/5
        if i % printtime == (printtime-1):
            running_loss /= printtime
            print('Step {}, Loss {:0.4f}'.format(i+1, running_loss))
            running_loss = 0
        if i == (num_trial-1):
            BN4_TestMLP.main(mlp_net)
    return mlp_net

import BN4_TestMLP
from datetime import datetime

def main():
    activity_name = ".\ANNModels\RNNActivityDict\_tmp_5_points"
    data,label = mk_traindata(activity_name,time_length = 5)

    mlp_net = MLPnet.MLPNet3(input_dim = 5)
    print(mlp_net)
    mlp_net = train_mlp(mlp_net,data,label)
    
    exportFlag = input('Export this model (y/n) :')
    if exportFlag == 'y':
        date = datetime.now().strftime("%m%d")
        filename = 'MLP_'+ date + "_" + input('Filename:'+'MLP_'+date+"_")
        with open( ".\\ANNModels\\MLP\\" + filename + '.pickle', mode='wb') as fo:
            pickle.dump(mlp_net, fo)        


if __name__ == "__main__":
    main()