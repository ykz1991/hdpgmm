import numpy as np
from scipy.stats import multivariate_normal
import pickle


def dict2mix(dic):
    dists = {}
    idx = 0
    weights = np.zeros(len(dic))
    for key in dic:
        if dic[key].n > 0:
            weights[idx] = 1.*dic[key].n
            mlt_norm = multivariate_normal(mean=dic[key].mean.A1, cov=dic[key].covar)
            dists[idx] = mlt_norm
            idx += 1
    weights = np.delete(weights, np.arange(len(dists), len(weights)))
    weights /= np.sum(weights)
    return weights, dists


def mixture_logpdf(x, weights, dists):
    from scipy.misc import logsumexp
    loglikelihoods = np.zeros(len(dists), dtype=np.float128)
    for key in dists:
        loglikelihoods[key] = np.log(weights[key]) + dists[key].logpdf(x)
    return logsumexp(loglikelihoods)


def all_loglike(X, weights, dists):
    tmp = 0.
    for x in X:
        tmp += mixture_logpdf(x, weights, dists)
    return tmp

'''
gaussian_hl = pickle.load(open('./results/2_model/gaussian_dists_hl.p', 'rb'))
gaussian_un = pickle.load(open('./results/2_model/gaussian_dists_un.p', 'rb'))
weights_hl, dists_hl = dict2mix(gaussian_hl)
weights_un, dists_un = dict2mix(gaussian_un)

idx = np.load('idx_ctu.npy')
feats = np.load('./features/feats.npy')
pH = np.load('pH.npy')
data = feats[idx, :, :]
pH = pH[idx]
data = data[:, :, [0, 1, 2, 3, 6]]
unhealthy = np.where(pH < 7.15)[0]
healthy = np.where(pH >= 7.15)[0]

test_ind_un = np.random.choice(unhealthy, 10)
test_ind_hl = np.random.choice(healthy, 10)
unhealthy_data = data[test_ind_un]
healthy_data = data[test_ind_hl]

un_loglike_un = np.exp(np.array([all_loglike(d, weights_un, dists_un) for d in unhealthy_data]))
un_loglike_hl = np.exp(np.array([all_loglike(d, weights_un, dists_un) for d in healthy_data]))
hl_loglike_hl = np.exp(np.array([all_loglike(d, weights_hl, dists_hl) for d in healthy_data]))
hl_loglike_un = np.exp(np.array([all_loglike(d, weights_hl, dists_hl) for d in unhealthy_data]))
print 'loglikelihood for healthy data in healthy model is ', hl_loglike_hl
print 'loglikelihood for healthy data in unhealthy model is ', un_loglike_hl
print 'loglikelihood for unhealthy data in unhealthy model is ', un_loglike_un
print 'loglikelihood for unhealthy data in healthy model is ', hl_loglike_un
'''