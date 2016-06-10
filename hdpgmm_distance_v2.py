# In this version, each time only one feature is used, in order to determined which feature is informative.

import numpy as np
import pickle
import os
from hdpgmm_class import Gaussian, GibbsSampler
np.seterr(divide='ignore')

# selecting model and feature parameters: segment length, alpha, gamma
segLens = [120, 160, 200]
params = [[100, 100]]
idx = np.load('idx_ctu.npy')
pH = np.load('pH.npy')
pH = pH[idx]
threshold = 7.15
unhealthy = np.where(pH < threshold)[0]
healthy = np.where(pH >= threshold)[0]
label = pH < threshold                  # 1 for unhealthy, 0 for healthy

for segLen in segLens:
    feats = np.load('./features/feats_time_freq_%d.npy' % segLen)
    print 'segment length is ', segLen, 'samples'
    meta_data = feats[idx, :, :]                 # use certain recordings according to idx
    for i in xrange(meta_data.shape[2]):
        data = meta_data[:, :, i:i+1]                  # each iteration only uses one feature
        unhealthy_data = data[unhealthy]
        healthy_data = data[healthy]

        folder = './results/2_model/feature_selection/feature_%i/' % i
        for param in params:
            alpha = param[0]
            gamma = param[1]
            print 'alpha is', alpha, ', gamma is', gamma
            subfolder = 'time_freq_%ds_feats_alpha_%d_gamma_%d/' % (segLen/4, alpha, gamma)
            directory = folder + subfolder
            if not os.path.isdir(directory):
                os.makedirs(directory)
            # train two HDPGMM models
            iteration = 50
            max_iteration = 150
            step = 10
            hdpgmm_un = GibbsSampler(snapshot_interval=10, compute_loglik=False)
            hdpgmm_hl = GibbsSampler(snapshot_interval=10, compute_loglik=False)

            hdpgmm_un._initialize(unhealthy_data, alpha=alpha, gamma=gamma)
            hdpgmm_hl._initialize(healthy_data, alpha=alpha, gamma=gamma)

            hdpgmm_un.sample(iteration)
            hdpgmm_un.pickle(directory, 'hdpgmm_un_%d-th_iter' % iteration)
            hdpgmm_hl.sample(iteration)
            hdpgmm_hl.pickle(directory, 'hdpgmm_hl_%d-th_iter' % iteration)

            for iter in xrange((max_iteration-iteration)/step):
                hdpgmm_un.sample(step)
                hdpgmm_un.pickle(directory, 'hdpgmm_un_%d-th_iter' % ((iter+1)*step+iteration))

                hdpgmm_hl.sample(step)
                hdpgmm_hl.pickle(directory, 'hdpgmm_hl_%d-th_iter' % ((iter+1)*step+iteration))
