import numpy as np
import math
import pickle
import os
from hdpgmm_class import Gaussian, GibbsSampler
from model_loglikelihood import dict2mix, all_loglike

segLen = 160
alpha = 1.
gamma = 1.
folder = 'time_freq_%ds_feats_alpha_%d_gamma_%d' % (segLen/4, alpha, gamma)
directory = './results/2_model/no_CV/%s/iteration_test/' % folder
for iter in xrange(20, 50):
    # hdpgmm_un = pickle.load(open(directory + 'hdpgmm_un_%d-th_iter' % iter))
    hdpgmm_hl = pickle.load(open(directory + 'hdpgmm_hl_%d-th_iter' % iter))
    weights_hl, dists_hl = dict2mix(hdpgmm_hl.params)
    # weights_un, dists_un = dict2mix(hdpgmm_un.params)
    print 'at %d-th iteration, ' % iter
    print 'log-likelihood is ' + str(hdpgmm_hl.get_logpdf())
    # for key in dists_hl:
    #    print 'weight %d is %f ' % (key, weights_hl[key]) + 'mean is ', dists_hl[key].mean
