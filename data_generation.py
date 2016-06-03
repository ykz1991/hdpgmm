import numpy as np
from hdpgmm_class import Gaussian, GibbsSampler
import matplotlib.pyplot as plt
import pickle
from model_loglikelihood import dict2mix, mixture_logpdf, all_loglike


def mix_multivariate_normal(weights, dists, n):
    counts = np.random.multinomial(n, weights)
    corpus = np.empty((0, len(dists[0].mean)))
    for i, c in enumerate(counts):
        tmp = np.random.multivariate_normal(dists[i].mean, dists[i].cov, c)
        corpus = np.concatenate((corpus, tmp), axis=0)
    return corpus


segLens = [40, 80, 120, 160, 200, 240]
params = [[1., 1.], [10., 10.], [100., 100], [1000., 1000.]]
for segLen in segLens:
    for param in params:
        alpha = param[0]
        gamma = param[1]
        directory = './results/2_model/no_CV_average/time_freq_%ds_feats_alpha_%d_gamma_%d/' % (segLen/4, alpha, gamma)
        num = 100000
        iter_start = 20
        iter_max = 100
        step_size = 10
        tnr = np.array([])
        tpr = np.array([])
        for iter in xrange(iter_start, iter_max, step_size):
            hdpgmm_un = pickle.load(open(directory + 'hdpgmm_un_%d-th_iter' % iter))
            hdpgmm_hl = pickle.load(open(directory + 'hdpgmm_hl_%d-th_iter' % iter))
            '''
            # plot log-likelihood against number of iterations
            plt.figure()
            x = np.arange(1, 21)
            plt.plot(x, hdpgmm_un.log_likelihoods, 'r')
            plt.xlim(1, 20)
            plt.xlabel('iteration')
            plt.ylabel('log-likelihood')
            plt.show()
            plt.figure()
            plt.plot(x, hdpgmm_hl.log_likelihoods, 'b')
            plt.xlim(1, 20)
            plt.xlabel('iteration')
            plt.ylabel('log-likelihood')
            plt.show()
            '''
            # transform class object to dictionary
            weights_hl, dists_hl = dict2mix(hdpgmm_hl.params)
            weights_un, dists_un = dict2mix(hdpgmm_un.params)
            # generate synthetic data
            corpus_hl = mix_multivariate_normal(weights_hl, dists_hl, num)
            corpus_un = mix_multivariate_normal(weights_un, dists_un, num)
            corpus = np.concatenate((corpus_hl, corpus_un), axis=0)
            '''
            print 'segment length is ', segLen
            print 'alpha is %d, gamma is %d ' % (alpha, gamma)
            print 'number of clusters in healthy mixture is ', len(weights_hl)
            print 'number of clusters in unhealthy mixture is ', len(weights_un)
            '''
            p0 = np.array([mixture_logpdf(data, weights_hl, dists_hl) for data in corpus])
            p1 = np.array([mixture_logpdf(data, weights_un, dists_un) for data in corpus])

            pred = (p0 < p1).astype(int)
            y = np.concatenate((np.zeros(num), np.ones(num)))
            tpr = np.append(tpr, 1.*np.sum(pred.astype(bool) & y.astype(bool))/np.sum(y))
            tnr = np.append(tnr, 1.*np.sum(~pred.astype(bool) & ~y.astype(bool))/(len(y) - np.sum(y)))

        # print 'segment length is %ds, alpha is %d, gamma is %d' % (segLen/4, alpha, gamma)
        print 'true positive rate is ', np.mean(tpr)
        print 'true negative rate is ', np.mean(tnr)
        print 'wra is ', np.mean(tpr)+np.mean(tnr)-1
