import numpy as np
import matplotlib.pyplot as plt

import gym  # package for RL environments
import neurogym as ngym

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import neurogym as ngym
import pickle


import sys
import ANNModels.RNNnet
import ANNModels.MLPnet
import ANNModels.RobustRNNnet as RobustRNNnet
sys.modules['RNNnet'] = ANNModels.RNNnet
sys.modules['MLPnet'] = ANNModels.MLPnet
import plotfunc
import ANNModels.SynthesisMLP

def create_RobustRNN(env,noise_Amplitude = False):
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    print("input_size :",input_size)
    print("output_size:",output_size)

    hidden_size = 64

    # RNNとMLPのweightをコピーする
    rnn_name = ".\ANNModels\RNN\_2024_09_12_15_14"
    file = open(rnn_name +'.pickle', mode='br')
    rnn_net = pickle.load(file)
    file.close()
    # MLPは合成モデルを用いる
    mlp_name = ".\ANNModels\MLP\MLP_" + "1210_1"
    file = open(mlp_name +'.pickle', mode='br')
    mlp_net1 = pickle.load(file)
    file.close()
    mlp_name = ".\ANNModels\MLP\MLP_" + "1115"
    file = open(mlp_name +'.pickle', mode='br')
    mlp_net2 = pickle.load(file)
    file.close()
    mlp_net = ANNModels.SynthesisMLP.SynthesisMLP(mlp_net1,mlp_net2)
    
    #feedback_flagでフィードバックの有無を切り替える
    net = RobustRNNnet.RobustRNNNet(
        input_size=input_size, 
        hidden_size=hidden_size,
        output_size=output_size, 
        dt=env.dt,
        rnn_net=rnn_net,
        mlp_net=mlp_net,
        feedback_flag = False,
        noise_flag = noise_Amplitude
        )
    return net

def test_rnn(net,num_trial,env,noise_Amplitude):
    activity_dict = []
    full_activity_dict = []
    trial_infos = {}
    acc = 0
    wa1 = 0
    count = 0
    rng = np.random.default_rng()
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        trial_infos[i] = env.trial.copy()
        #入力にノイズを加える(加えない場合はnoise_Amplitudeを0にする)
        ob.T[1][10:40] += rng.normal(0, noise_Amplitude, 30)
        #input_data, label_dataをプロットする
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
        if trial_infos[i]['ground_truth']==torch.argmax(action_pred[-1]).numpy():
            acc += 1
        elif trial_infos[i]["ground_truth"] == 1:
            wa1 += 1
        count += 1
    print(noise_Amplitude,":",acc/count)  
    #正解ラベルが1、回答が2の誤答数
    print("WA1:", wa1) 
    #正解ラベルが2、回答が1の誤答数
    print("WA2:", (count - acc - wa1))

    return activity_dict,full_activity_dict,trial_infos

def main():
    task = 'DelayComparison-v0'
    kwargs = {'timing': {'delay': ('constant', 3000)}}
    env = gym.make(task, **kwargs)
    env.reset(no_step=True)
    env.timing

    
    for noise_Amplitude in [0.20]:
        #RNNモデルの作成。unitに摂動を加える際は第二引数をnoise_Amplitudeに変更する
        net = create_RobustRNN(env,False)
        num_trial = 200
        activity_dict, full_activity_dict, trial_infos = test_rnn(net,num_trial,env,noise_Amplitude)
    
    #アクティビティ(64次元)、PCAの成分(2次元)をプロットする
    plotfunc.plot_activity_Individual(trial_infos,num_trial,activity_dict,length=30)
    pca = plotfunc.create_pca(activity_dict,num_trial)
    plotfunc.plot_activity_pca(trial_infos,num_trial,activity_dict,pca)
    plt.show()


if __name__ == "__main__":
    main()
