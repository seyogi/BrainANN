import random
import gym  # package for RL environments
import neurogym as ngym
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def create_train_data():
    _task = 'DelayComparison-v0'
    _timing = {'delay': ('choice', [3000]),
            'response': ('constant', 500)
            }
    _kwargs = {'dt': 100, 'timing': _timing}
    _seq_len = 100

    _dataset = ngym.Dataset(_task, env_kwargs=_kwargs, batch_size=16, seq_len=_seq_len)
    return _dataset

def main():
  dataset = create_train_data()
  env = dataset.env
  env.new_trial()
  ob, gt = env.ob, env.gt
  trial_infos = env.trial.copy()
  Index = 2
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
  print(gt)
  print(labels_np.shape)
  ax1.plot(ob.T[0], label="fixation")
  ax1.plot(ob.T[1], label="stimulus")
  ax1.legend(loc=2)
  ax2.plot(gt, color="g", label="Ground truth")
  ax2.legend(loc=2)
  plt.show()
  """
  date = datetime.now().strftime(".\CreateDelayTask\_%Y_%m_%d_%H_%M_%S")
  with open( date + '.pickle', mode='wb') as fo:
    pickle.dump(dataset, fo)
  """

if __name__ == "__main__":
    main()