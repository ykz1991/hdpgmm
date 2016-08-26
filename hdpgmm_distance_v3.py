# In this version, the Gibbs sampling results are averaged across different iterations, determined by variables
# iteration, max_iteration and step.
# What's new: gamma priors are given to the concentration parameters

import numpy as np
import os
from hdpgmm_class_v2 import GibbsSampler
from sklearn.decomposition import PCA
from sklearn import preprocessing

# selecting model and feature parameters: segment length, alpha, gamma
for run in xrange(2):
    segLens = [40, 80, 120]
    idx = np.load('./index/idx_705.npy')
    pH = np.load('pH.npy')
    pH = pH[idx]
    threshold = 7.05
    unhealthy = np.where(pH <= threshold)[0]
    healthy = np.where(pH > threshold)[0]
    label = pH <= threshold                  # 1 for unhealthy, 0 for healthy
    q = 3

    for segLen in segLens:
        feats = np.load('./features/feats_time_freq_%d.npy' % segLen)
        print 'segment length is', segLen, 'samples,', 'dimension is ', q
        data = feats[idx, :, :]                 # use certain recordings according to idx

        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        pca = PCA(n_components=q)
        shape = data.shape
        data_reshape = np.reshape(data, (shape[0]*shape[1], shape[2]))
        data_reshape_scaled = scaler.fit_transform(data_reshape)
        data_pca_reshape = pca.fit_transform(data_reshape_scaled)
        data_pca = np.reshape(data_pca_reshape, (shape[0], shape[1], q))

        unhealthy_data = data_pca[unhealthy]
        healthy_data = data_pca[healthy]

        folder = 'time_freq_dim%d_%ds_feats_%dst_run' % (q, segLen/4, (run+1))
        directory = './results/2_model/no_CV_average_hyper_param_pca_scaled/%s/' % folder
        if not os.path.isdir(directory):
            os.makedirs(directory)
        # train two HDPGMM models
        iteration = 60
        max_iteration = 80
        step = 10
        hdpgmm_un = GibbsSampler(snapshot_interval=10, compute_loglik=True)
        hdpgmm_hl = GibbsSampler(snapshot_interval=10, compute_loglik=True)

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
