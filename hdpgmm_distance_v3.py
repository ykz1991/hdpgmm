# In this version, the Gibbs sampling results are averaged across different iterations, determined by variables
# iteration, max_iteration and step.
# What's new: gamma priors are given to the concentration parameters

import numpy as np
import pickle
import os
from hdpgmm_class_v2 import Gaussian, GibbsSampler

# selecting model and feature parameters: segment length, alpha, gamma
segLens = [40, 80, 120, 160, 200, 240]
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
    data = feats[idx, :, :]                 # use certain recordings according to idx
    # data = data[:, :, [0, 1, 2, 3, 6, 7]]   # ARX coefficients, std, mean of rr interval
    # data = data[:, :, [0, 3, 5, 6, 7, 8]]   # mean, sti, lti, poincare
    unhealthy_data = data[unhealthy]
    healthy_data = data[healthy]

    folder = 'time_freq_%ds_feats' % (segLen/4)
    directory = './results/2_model/no_CV_average_hyper_param/%s/' % folder
    if not os.path.isdir(directory):
        os.makedirs(directory)
    # train two HDPGMM models
    iteration = 30
    max_iteration = 100
    step = 10
    hdpgmm_un = GibbsSampler(snapshot_interval=5, compute_loglik=False)
    hdpgmm_hl = GibbsSampler(snapshot_interval=5, compute_loglik=False)

    hdpgmm_un._initialize(unhealthy_data)
    hdpgmm_hl._initialize(healthy_data)

    hdpgmm_un.sample(iteration)
    hdpgmm_un.pickle(directory, 'hdpgmm_un_%d-th_iter' % iteration)
    hdpgmm_hl.sample(iteration)
    hdpgmm_hl.pickle(directory, 'hdpgmm_hl_%d-th_iter' % iteration)

    for iter in xrange((max_iteration-iteration)/step):
        hdpgmm_un.sample(step)
        hdpgmm_un.pickle(directory, 'hdpgmm_un_%d-th_iter' % ((iter+1)*step+iteration))

        hdpgmm_hl.sample(step)
        hdpgmm_hl.pickle(directory, 'hdpgmm_hl_%d-th_iter' % ((iter+1)*step+iteration))
