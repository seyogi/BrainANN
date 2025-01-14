import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def create_pca(_activity_dict,num_trial):
    _activity = np.concatenate(list(_activity_dict[i] for i in range(num_trial)), axis=0)
    print('Shape of the neural activity: (Time points, Neurons): ', _activity.shape)
    _pca = PCA(n_components=2)
    _pca.fit(_activity)
    return _pca

def plot_activity(trial_infos,activity_dict,full_activity_dict,hidden_size=64,Trial_index = 0):
    # Print trial informations
    Trial_index = 0
    print('\nTrial ', Trial_index, trial_infos[Trial_index])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(hidden_size):
        ax1.plot(activity_dict[Trial_index].T[i])
    for i in range(hidden_size):
        ax2.plot(full_activity_dict[Trial_index].T[i])

def plot_activity_Individual(trial_infos,num_trial,activity_dict,hidden_size=64,length=30):
    fig2 = plt.figure(figsize = (15,5))
    _activity = np.concatenate(list(activity_dict[i] for i in range(num_trial)), axis=0)
    if num_trial > 100:
        num_trial = 100
    for i in range(hidden_size):
        ax = fig2.add_subplot(8, 8, i+1)
        tmp = _activity[:,i]
        for j in range(num_trial):
            trial = trial_infos[j]
            color = '#FFCCCC' if trial['v1'] <= 14 else '#FF9999' if trial['v1'] <= 20 else '#FF6666' if trial['v1'] < 30 else '#FF0000'
            #ax.set_ylim(-0,1)
            ax.plot(tmp[length*j:length*(j+1)], color=color,alpha=1)
            ax.axis("off")

def plot_activity_pca(trial_infos,num_trial,activity_dict,pca):
    fig, (ax1) = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(7, 7))
    if num_trial > 100:
        num_trial = 100
    for i in range(num_trial):
        trial = trial_infos[i]
        activity_pc = pca.transform(activity_dict[i])
        color = '#FFCCCC' if trial['v1'] <= 14 else '#FF9999' if trial['v1'] <= 20 else '#FF6666' if trial['v1'] < 30 else '#FF0000'
        
        _ = ax1.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)

    for i in range(num_trial):
        activity_pc = pca.transform(activity_dict[i])
        trial = trial_infos[i]
        ax1.plot(activity_pc[0, 0], activity_pc[0, 1], '^', color="green", alpha=1)
        color = '#CCCCFF' if trial['v1'] <= 14 else '#9999FF' if trial['v1'] <= 20 else '#6666FF' if trial['v1'] < 30 else '#0000FF'
        ax1.plot(activity_pc[-1, 0], activity_pc[-1, 1], 'x', color=color)

    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    return pca