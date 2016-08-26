import numpy as np
import pickle
from model_loglikelihood import dict2mix, mixture_logpdf
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_gm(obj, ax, indicator=0):
    weights, dists = dict2mix(obj.params)
    c = ['r', 'g', 'b', 'c', 'y']
    h = ['', '/']
    for n, weight in enumerate(weights):
        if weight < 0.01:
            continue
        v, w = np.linalg.eigh(dists[n].cov)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi
        v *= 2

        ell = Ellipse(dists[n].mean, v[0], v[1], 180 + angle, color=c[n % 5], hatch=h[indicator])
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(weight)
        ax.add_artist(ell)
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 1)


segLen = 80
q = 2
directory0 = './results/2_model/no_CV_average_hyper_param_pca_scaled/time_freq_dim%d_%ds_feats_2st_run/' % (q, segLen/4)
directory1 = './results/2_model/no_CV_average_hyper_param_pca_scaled/time_freq_dim%d_%ds_feats_1st_run/' % (q, segLen/4)
directory2 = './results/2_model/no_CV_average_hyper_param_pca_scaled/time_freq_dim%d_%ds_feats_3st_run/' % (q, segLen/4)

iter0 = 60
iter1 = 70
iter2 = 80
hdpgmm_un_0 = pickle.load(open(directory1 + 'hdpgmm_un_%d-th_iter' % iter0))
hdpgmm_un_1 = pickle.load(open(directory1 + 'hdpgmm_un_%d-th_iter' % iter1))
hdpgmm_un_2 = pickle.load(open(directory1 + 'hdpgmm_un_%d-th_iter' % iter2))

hdpgmm_hl_0 = pickle.load(open(directory2 + 'hdpgmm_hl_%d-th_iter' % iter0))
hdpgmm_hl_1 = pickle.load(open(directory2 + 'hdpgmm_hl_%d-th_iter' % iter1))
hdpgmm_hl_2 = pickle.load(open(directory2 + 'hdpgmm_hl_%d-th_iter' % iter2))

hdpgmm_un_0_2 = pickle.load(open(directory0 + 'hdpgmm_un_%d-th_iter' % iter0))
hdpgmm_un_1_2 = pickle.load(open(directory0 + 'hdpgmm_un_%d-th_iter' % iter1))
hdpgmm_un_2_2 = pickle.load(open(directory0 + 'hdpgmm_un_%d-th_iter' % iter2))

hdpgmm_un_0_3 = pickle.load(open(directory2 + 'hdpgmm_un_%d-th_iter' % iter0))
hdpgmm_un_1_3 = pickle.load(open(directory2 + 'hdpgmm_un_%d-th_iter' % iter1))
hdpgmm_un_2_3 = pickle.load(open(directory2 + 'hdpgmm_un_%d-th_iter' % iter2))

print 'loading finished.'


fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_un_0, ax)

fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_un_0_3, ax, 1)

fig = plt.figure(1)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_un_1, ax)

fig = plt.figure(1)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_un_1_3, ax, 1)

fig = plt.figure(2)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_un_2, ax)

fig = plt.figure(2)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_un_2_3, ax, 1)

plt.show()

'''
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_un_0, ax)

fig = plt.figure(1)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_un_1, ax)

fig = plt.figure(2)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_un_2, ax)

fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_hl_0, ax, True)

fig = plt.figure(1)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_hl_1, ax, True)

fig = plt.figure(2)
ax = fig.add_subplot(111, aspect='equal')
plot_gm(hdpgmm_hl_2, ax, True)

plt.show()
'''