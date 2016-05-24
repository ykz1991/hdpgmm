import numpy as np
from hdpgmm_class import Gaussian, GibbsSampler
import pickle
from model_loglikelihood import dict2mix, mixture_logpdf, all_loglike


def mix_multivariate_normal(weights, dists, n):
    counts = np.random.multinomial(n, weights)
    corpus = np.empty((0, len(dists[0].mean)))
    for i, c in enumerate(counts):
        tmp = np.random.multivariate_normal(dists[i].mean, dists[i].cov, c)
        corpus = np.concatenate((corpus, tmp), axis=0)
    return corpus


directory = './results/2_model/no_CV/time_freq_30s_feats_alpha_5_gamma_10/'
hdpgmm_un = pickle.load(open(directory + 'hdpgmm_un', 'rb'))
hdpgmm_hl = pickle.load(open(directory + 'hdpgmm_hl', 'rb'))
# transform class object to dictionary
weights_hl, dists_hl = dict2mix(hdpgmm_hl.params)
weights_un, dists_un = dict2mix(hdpgmm_un.params)

num = 500000
corpus_hl = mix_multivariate_normal(weights_hl, dists_hl, num)
corpus_un = mix_multivariate_normal(weights_un, dists_un, num)
corpus = np.concatenate((corpus_hl, corpus_un), axis=0)

p0 = np.array([mixture_logpdf(data, weights_hl, dists_hl) for data in corpus])
p1 = np.array([mixture_logpdf(data, weights_un, dists_un) for data in corpus])
pred = (p0 < p1).astype(int)
y = np.concatenate((np.zeros(num), np.ones(num)))
tpr = 1.*np.sum(pred.astype(bool) & y.astype(bool))/np.sum(y)
tnr = 1.*np.sum(~pred.astype(bool) & ~y.astype(bool))/(len(y) - np.sum(y))
print 'true positive rate is ', tpr
print 'true negative rate is ', tnr
