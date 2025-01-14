import numpy as np
import matplotlib.pyplot as plt

x   = [0.00,0.02,0.04,0.06,0.08, 0.10,0.12,0.14,0.16,0.18 ,0.20,0.22,0.24,0.26,0.28, 0.30,0.32,0.34,0.36,0.38, 0.40]
#noise to input
RNN = [0.930,0.908,0.898,0.928,0.874, 0.846,0.820,0.746,0.794,0.742, 0.728,0.738,0.700,0.678,0.678, 0.688,0.632,0.658,0.626,0.632, 0.644]
MLP = [0.978,0.978,0.972,0.970,0.962, 0.952,0.942,0.914,0.906,0.904, 0.860,0.834,0.812,0.838,0.786, 0.762,0.736,0.736,0.664,0.668, 0.650]
kalman = [0.876,0.888,0.882,0.832,0.8,0.782,0.716,0.732,0.706,0.652, 0.660,0.684,0.622,0.586,0.616, 0.590,0.570,0.550,0.540,0.580, 0.556]
#noise to unit
RNN2 = [0.924,0.914,0.908,0.838,0.814 ,0.794,0.766,0.704,0.744,0.654, 0.676,0.616,0.644,0.600,0.570, 0.552,0.544,0.462,0.502,0.454, 0.390]
MLP2 = [0.976,0.962,0.960,0.952,0.944 ,0.898,0.894,0.856,0.848,0.824, 0.796,0.750,0.782,0.718,0.714, 0.674,0.614,0.620,0.578,0.572 ,0.548]

fig = plt.figure(figsize = (6,3))
#fig.set_facecolor((0.9, 0.9, 0.9, 0.5))

ax = fig.add_subplot(1, 1, 1)
ax.title.set_text('noise to input')
ax.set_ylim(0.4, 1)
ax.plot(x,RNN, color="#FF0000",alpha=1, label="RNN only")
ax.plot(x,MLP, color="#0000FF",alpha=1, label="RNN + MLP")
ax.plot(x,kalman, color="#00FF00",alpha=1, label="Linear Kalman")
p = plt.hlines([0.5], 0, 0.40, "black", linestyles='dashed') 
p = plt.xlabel('noise power')
p = plt.ylabel('accuracy')
ax.legend(loc=1)

'''
ax2 = fig.add_subplot(2, 1, 2)
ax2.title.set_text('noise to unit')
ax2.set_ylim(0.4, 1)
ax2.plot(x,RNN2, color="#FF0000",alpha=1, label="Model A")
ax2.plot(x,MLP2, color="#0000FF",alpha=1, label="Model B")
p = plt.hlines([0.5], 0, 0.40, "black", linestyles='dashed') 
p = plt.xlabel('noise power')
p = plt.ylabel('accuracy')
ax2.legend(loc=1)
'''

plt.subplots_adjust(wspace=0.4,hspace=1)
plt.show()