import numpy as np
import matplotlib.pyplot as plt
import pickle
from hdpgmm_class_v2 import GibbsSampler

n_samples0 = 150   # Number of samples per component
n_samples1 = 150
n_samples2 = 100
frac0 = [.2, .5, .3]
frac1 = [.5, .3, .2]
frac2 = [.6, .4]
# 2, 2-dimensional Gaussian
mean = [[0., 0.], [3., 1.], [1., -3.], [-3., -3.], [-3., 1.]]
cov = [[.5, 0.], [0., .5]]
seg = 3
data = np.zeros((seg, n_samples0+n_samples1+n_samples2, 2))
true_c = {}
directory = './test/'

color_iter = ['r', 'g', 'b', 'c', 'm']
marker_iter =['+', 'v', 'o', 's', '*']

for ct in range(seg):
    if ct == 0:
        Xdata0 = np.r_[np.random.multivariate_normal(mean[:][0], cov, 30),
                       np.random.multivariate_normal(mean[:][1], cov, 75),
                       np.random.multivariate_normal(mean[:][2], cov, 45)]
        Xdata1 = np.r_[np.random.multivariate_normal(mean[:][0], cov, 75),
                       np.random.multivariate_normal(mean[:][3], cov, 45),
                       np.random.multivariate_normal(mean[:][4], cov, 30)]
        Xdata2 = np.r_[np.random.multivariate_normal(mean[:][1], cov, 60),
                       np.random.multivariate_normal(mean[:][4], cov, 40)]
    if ct == 1:
        Xdata0 = np.r_[np.random.multivariate_normal(mean[:][0], cov, 80),
                       np.random.multivariate_normal(mean[:][1], cov, 20),
                       np.random.multivariate_normal(mean[:][2], cov, 50)]
        Xdata1 = np.r_[np.random.multivariate_normal(mean[:][0], cov, 50),
                       np.random.multivariate_normal(mean[:][3], cov, 50),
                       np.random.multivariate_normal(mean[:][4], cov, 50)]
        Xdata2 = np.r_[np.random.multivariate_normal(mean[:][1], cov, 80),
                       np.random.multivariate_normal(mean[:][4], cov, 20)]
    if ct == 2:
        Xdata0 = np.r_[np.random.multivariate_normal(mean[:][0], cov, 30),
                       np.random.multivariate_normal(mean[:][1], cov, 100),
                       np.random.multivariate_normal(mean[:][2], cov, 20)]
        Xdata1 = np.r_[np.random.multivariate_normal(mean[:][0], cov, 75),
                       np.random.multivariate_normal(mean[:][3], cov, 75)]
        Xdata2 = np.r_[np.random.multivariate_normal(mean[:][1], cov, 20),
                       np.random.multivariate_normal(mean[:][4], cov, 80)]
    ind0 = np.random.permutation(len(Xdata0))
    ind1 = np.random.permutation(len(Xdata1))
    ind2 = np.random.permutation(len(Xdata2))
    X0 = Xdata0[ind0]
    X1 = Xdata1[ind1]
    X2 = Xdata2[ind2]
    X = np.r_[X0, X1, X2]
    data[ct] = X
    ind = np.r_[ind0, ind1, ind2]
    true_cluster = []
    if ct == 0:
        for i in ind0:
            if i < 30:
                true_cluster.append(0)
            elif i < 105:
                true_cluster.append(1)
            else:
                true_cluster.append(2)
        for i in ind1:
            if i < 75:
                true_cluster.append(0)
            elif i < 120:
                true_cluster.append(3)
            else:
                true_cluster.append(4)
        for i in ind2:
            if i < 60:
                true_cluster.append(1)
            else:
                true_cluster.append(4)
    if ct == 1:
        for i in ind0:
            if i < 80:
                true_cluster.append(0)
            elif i < 100:
                true_cluster.append(1)
            else:
                true_cluster.append(2)
        for i in ind1:
            if i < 50:
                true_cluster.append(0)
            elif i < 100:
                true_cluster.append(3)
            else:
                true_cluster.append(4)
        for i in ind2:
            if i < 80:
                true_cluster.append(1)
            else:
                true_cluster.append(4)
    if ct == 2:
        for i in ind0:
            if i < 30:
                true_cluster.append(0)
            elif i < 130:
                true_cluster.append(1)
            else:
                true_cluster.append(2)
        for i in ind1:
            if i < 75:
                true_cluster.append(0)
            else:
                true_cluster.append(3)
        for i in ind2:
            if i < 20:
                true_cluster.append(1)
            else:
                true_cluster.append(4)
    true_c[ct] = true_cluster
    plt.figure()
    for j in range(len(mean)):
        plt.scatter(X[np.array(true_cluster) == j, 0], X[np.array(true_cluster) == j, 1], marker=marker_iter[j], color=color_iter[j])
        plt.title('data distribution in time series %i' % (ct+1))
    # pl.show()
    plt.savefig(directory + 'data_dist_%i' % ct)
np.save(directory + 'data', data)
pickle.dump(true_c, open(directory + 'true_c', 'wb'))


sampler = GibbsSampler(snapshot_interval=10, compute_loglik=True)
sampler._initialize(data=data)
sampler.sample(100)
kdt = sampler._k_dt
tdv = sampler._t_dv
pickle.dump(kdt, open(directory + 'kdt', 'wb'))
pickle.dump(tdv, open(directory + 'tdv', 'wb'))
for s in xrange(seg):
    plt.figure()
    X = data[s]
    for j in xrange(sampler._K):
        plt.scatter(X[kdt[s][tdv[s]] == j, 0], X[kdt[s][tdv[s]] == j, 1], marker=marker_iter[j % 5], color=color_iter[j % 5])
        plt.title('data inference result in time series %i' % (s + 1))
    plt.savefig(directory + 'data_inference_%i' % s)
