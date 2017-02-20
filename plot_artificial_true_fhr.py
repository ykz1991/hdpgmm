import numpy as np
import matplotlib.pyplot as plt
import os

sim_fhr_h = np.load('../../data/simulated/fhr_h.npy')
sim_fhr_u = np.load('../../data/simulated/fhr_u.npy')
real_fhr_h = np.load('../../data/simulated/fhr_t_h.npy')
real_fhr_u = np.load('../../data/simulated/fhr_t_u.npy')

index = [3, 5, 8]
chosen_fhr = np.concatenate([real_fhr_h[1:4], sim_fhr_h[1:4], real_fhr_u[1:4], sim_fhr_u[1:4], real_fhr_u[10:12],
                             sim_fhr_h[10:12], sim_fhr_u[10:12], real_fhr_h[10:12]])
chosen_fhr_10min = chosen_fhr[:, -2400:]

for i in xrange(20):
    x = np.arange(2400)/4.
    plt.figure(figsize=(12, 4.5))
    plt.plot(x, chosen_fhr_10min[i])
    plt.xlabel('time (sec)')
    plt.ylabel('FHR')
    plt.title('Figure %d' %(i+1))
    plt.savefig('./figure/fhr_%d' % (i+1))

# 20 tracings are R R R S S S R R R S S S R R S S S S R R
