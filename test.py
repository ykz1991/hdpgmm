import numpy as np
import matplotlib.pyplot as plt
import pickle
from hdpgmm_class_v2 import GibbsSampler

# 2, 2-dimensional Gaussian
mean = [[0., 0.], [3., 1.], [1., -3.], [-3., -3.]]
cov = [[.5, 0.], [0., .5]]
seg = 10
directory = './test/'

color_iter = ['r', 'g', 'b', 'm', 'c']
marker_iter =['+', 'v', 'o', 's', '*']

num = 100
data = np.zeros((seg, num, 2))
true_cluster = np.zeros((seg, num))
prob = np.random.dirichlet(np.ones(len(mean)), seg)
for s in xrange(seg):
    for n in xrange(num):
        tmp_c = np.random.choice(len(mean), 1, p=prob[s])[0]
        tmp_d = np.random.multivariate_normal(mean[tmp_c], cov)
        data[s, n, :] = tmp_d
        true_cluster[s, n] = tmp_c
    plt.figure()
    X = data[s]
    cluster = true_cluster[s]
    for j in range(len(mean)):
        plt.scatter(X[np.array(cluster) == j, 0], X[np.array(cluster) == j, 1], marker=marker_iter[j], color=color_iter[j])
        plt.title('data distribution in time series %i' % (s+1))
    # pl.show()
    plt.savefig(directory + 'data_dist_%i' % s)
np.save(directory + 'data', data)
np.save(directory + 'true_cluster', true_cluster)


sampler = GibbsSampler(snapshot_interval=10, compute_loglik=True)
sampler.initialize(data=data)
snap_interval = 50
iteration = 50
for tmp in xrange(iteration/snap_interval):
    sampler.sample(snap_interval)
    kdt = sampler._k_dt
    tdv = sampler._t_dv
    # pickle.dump(kdt, open(directory + 'kdt', 'wb'))
    # pickle.dump(tdv, open(directory + 'tdv', 'wb'))
    for s in xrange(seg):
        plt.figure()
        X = data[s]
        for j in xrange(sampler._K):
            plt.scatter(X[kdt[s][tdv[s]] == j, 0], X[kdt[s][tdv[s]] == j, 1], marker=marker_iter[j % 5], color=color_iter[j % 5])
            plt.title('data inference result in time series %i' % (s + 1))
        plt.savefig(directory + 'data_inference_%i' % s)
