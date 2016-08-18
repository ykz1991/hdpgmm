# In this version, the Gibbs sampling results are averaged across different iterations, determined by variables
# iteration, max_iteration and step.
# What's new: use data after PCA (dim=2 or 3)

import numpy as np
import os
from hdpgmm_class_v2 import GibbsSampler

# selecting model and feature parameters: segment length
for run in xrange(1):
    segLens = [200, 220, 240]

    for segLen in segLens:
        unhealthy_data = np.load('./pca_data/2dim/unhealthy_data_pca_%ds.npy' % (segLen/4))
        healthy_data = np.load('./pca_data/2dim/healthy_data_pca_%ds.npy' % (segLen/4))

        folder = 'time_freq_%ds_feats_%dth_run' % (segLen/4, (run+4))
        directory = './results/2_model/no_CV_average_hyper_param_pca/%s/' % folder
        if not os.path.isdir(directory):
            os.makedirs(directory)
        # train two HDPGMM models
        iteration = 50
        max_iteration = 100
        step = 10
        hdpgmm_un = GibbsSampler(snapshot_interval=10, compute_loglik=False)
        hdpgmm_hl = GibbsSampler(snapshot_interval=10, compute_loglik=False)

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

            print '%d-th iteration finished.' % ((iter + 1) * step + iteration)
