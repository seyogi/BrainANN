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
import copy

def activity_normalization(_activity_dict,num_trial,hidden_size = 64):
    _activity_dict2 = copy.deepcopy(_activity_dict)
    _tmp_max = np.zeros(hidden_size)
    _tmp_min = np.ones(hidden_size)
    #各ニューロンActivityの最大値を求める
    for i in range(num_trial):
        for j in range(hidden_size):
            tmp = max(_activity_dict[i][:,j])
            tmp_s = min(_activity_dict[i][:,j])
            if _tmp_max[j] < tmp:
                _tmp_max[j] = tmp
            if _tmp_min[j] > tmp_s:
                _tmp_min[j] = tmp_s
    #各ニューロンActivityの最大値で割る
    for i in range(num_trial):
        for j in range(hidden_size):
            tmp = max(_activity_dict[i][:,j])
            if tmp != 0:
                _activity_dict2[i][:,j] = _activity_dict[i][:,j] / _tmp_max[j]
    return _activity_dict2

def create_activity_dict(_activity_dict,_seq_len,num_trial,_time_length = 5):
    _tmp_dict = {}
    for i in range(num_trial):
        for t in range(_seq_len - _time_length): #0~40
            _tmp_dict[i*(_seq_len-_time_length)+t] = {"data":np.concatenate([_activity_dict[i][t+j] for j in range(_time_length)], axis = 0) ,"label":_activity_dict[i][t+_time_length]}
    return _tmp_dict

def mk_activity_dicts(_activity_dict,_seq_len,num_trial,time_length):
    _tmp_dict = create_activity_dict(_activity_dict,_seq_len,num_trial,_time_length = time_length)
    return _tmp_dict, _activity_dict