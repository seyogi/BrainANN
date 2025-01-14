import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import ANNModels.RNNnet
import plotfunc
import create_traindata

import gym  # package for RL environments
import neurogym as ngym
from sklearn.decomposition import PCA

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import ANNModels

def test_rnn(net,num_trial,env):
    activity_dict = []
    full_activity_dict = []
    trial_infos = {}
    acc = 0
    count = 0
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        trial_infos[i] = env.trial.copy()
        if i<1:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(5, 5))
            fig.suptitle(trial_infos[i])
            ax1.plot(ob.T[0], label="fixation")
            ax1.plot(ob.T[1], label="stimulus")
            ax1.legend(loc=2)
            ax2.plot(gt, label="Ground truth")
            ax2.legend(loc=2)
        inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
        action_pred, rnn_activity = net(inputs)
        rnn_activity = rnn_activity[:, 0, :].detach().numpy()
        activity_dict.append(rnn_activity[env.start_ind['delay']:env.end_ind['delay']])
        full_activity_dict.append(rnn_activity)
        #print(trial_infos[i]['ground_truth'],torch.argmax(action_pred[-1]))
        #print(action_pred)
        if trial_infos[i]['ground_truth']==torch.argmax(action_pred[-1]).numpy():
            acc += 1
        count += 1
    print(acc/count)  
    
    return activity_dict,full_activity_dict,trial_infos

def main():
    sys.modules['RNNnet'] = ANNModels.RNNnet
    #road rnn
    rnn_name = ".\ANNModels\RNN\_2024_09_12_15_14"
    file = open(rnn_name +'.pickle', mode='br')
    net = pickle.load(file)
    file.close()
    hidden_size = net.rnn.hidden_size

    task = 'DelayComparison-v0'
    kwargs = {'timing': {'delay': ('constant', 3000)}}
    env = gym.make(task, **kwargs)
    env.reset(no_step=True)
    env.timing

    num_trial = 2000
    activity_dict, full_activity_dict, trial_infos = test_rnn(net,num_trial,env)

    plotfunc.plot_activity(trial_infos,activity_dict,full_activity_dict)
    plotfunc.plot_activity_Individual(trial_infos,num_trial,activity_dict)
    pca = plotfunc.create_pca(activity_dict,num_trial)
    plotfunc.plot_activity_pca(trial_infos,num_trial,activity_dict,pca)
    
    #pcaのプロット
    
    fig3, (ax3)  = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(7, 7))
    Trial_index = 5
    activity_pc = pca.transform(activity_dict[Trial_index])
    ax3.plot(activity_pc)
    plt.show()
    

    time_length = 1
    new_activity_dict, full_activity_small_dict  = create_traindata.mk_activity_dicts(full_activity_dict,46,num_trial,time_length)
    
    exportFlag = input('Export this model (y/n) :')
    if exportFlag == 'y':
        filename = ".\ANNModels\RNNActivityDict\_"+ time_length +"_points"
        with open( filename + '.pickle', mode='wb') as fo:
            pickle.dump([new_activity_dict,full_activity_small_dict,trial_infos,pca], fo)


if __name__ == "__main__":
    main()