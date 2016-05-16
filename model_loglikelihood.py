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

