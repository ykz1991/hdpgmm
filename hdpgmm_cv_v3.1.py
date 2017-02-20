import numpy as np
import pickle
import os
from sklearn.model_selection import StratifiedKFold
from model_loglikelihood import dict2mix, all_loglike
from hdpgmm_class_v2 import GibbsSampler
from sklearn.decomposition import PCA
from sklearn import preprocessing

np.seterr(divide='ignore')

# This version investigates jointly processing FHR and UA signals

idx = np.load('./index/idx_705_unbalanced.npy')
pH = np.load('pH.npy')
pH = pH[idx]
threshold = 7.05
unhealthy = np.where(pH <= threshold)[0]
healthy = np.where(pH > threshold)[0]
label = (pH <= threshold).astype(int)                  # 1 for unhealthy, 0 for healthy
n_fold = 5
skf = StratifiedKFold(n_splits=n_fold)

segLens = [40, 80, 120]
qs = [2, 3, 4]                                         # dimension after PCA
# Number of iteration in sampling
iter_start = 60
iter_stop = 90
iter_step = 10
step = (iter_stop - iter_start) / iter_step
for q in qs:
    for segLen in segLens:
        feats = np.load('./features/featsFHR_time_freq_%d.npy' % segLen)
        print 'segment length is %ds' % (segLen / 4), ', q is %d' % q
        data = feats[idx, :, :]                 # use certain recordings according to idx

        folder = 'time_freq_dim%d_%ds_FHR' % (q, segLen/4)
        directory = './results/2_model/CV_hyper_param_pca_scaled_7.05_unbalanced/%s/' % folder
        if not os.path.isdir(directory):
            os.makedirs(directory)

        tmp_accuracy = np.zeros(n_fold)
        tmp_tnr = np.zeros(n_fold)
        tmp_tpr = np.zeros(n_fold)
        run = 0
        CV_idx = {}

        for train, test in skf.split(data, label):
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            pca = PCA(n_components=q)
            shape_train = data[train].shape
            data_train_reshape = np.reshape(data[train], (shape_train[0] * shape_train[1], shape_train[2]))
            shape_test = data[test].shape
            data_test_reshape = np.reshape(data[test], (shape_test[0] * shape_test[1], shape_test[2]))

            data_train_pca_reshape_scaled = scaler.fit_transform(data_train_reshape)
            data_train_pca_reshape = pca.fit_transform(data_train_pca_reshape_scaled)

            data_test_pca_reshape_scaled = scaler.transform(data_test_reshape)
            data_test_pca_reshape = pca.transform(data_test_pca_reshape_scaled)

            data_train_pca = np.reshape(data_train_pca_reshape, (shape_train[0], shape_train[1], q))
            data_test_pca = np.reshape(data_test_pca_reshape, (shape_test[0], shape_test[1], q))
            # train two HDPGMM models
            # initialization
            hdpgmm_un = GibbsSampler(snapshot_interval=20)
            hdpgmm_hl = GibbsSampler(snapshot_interval=20)
            hdpgmm_un.initialize(data_train_pca[pH[train] <= threshold])
            hdpgmm_hl.initialize(data_train_pca[pH[train] > threshold])

            hdpgmm_un.sample(iter_start)
            hdpgmm_hl.sample(iter_start)
            hdpgmm_un.pickle(directory, 'hdpgmm_un_%d_run_%d-th_iter' % (run+1, iter_start))
            hdpgmm_hl.pickle(directory, 'hdpgmm_hl_%d_run_%d-th_iter' % (run+1, iter_start))
            p0_all = np.empty((len(test), step))
            p1_all = np.empty((len(test), step))
            # average sampling results from iter_start to iter_stop for every iter_step iterations
            for iter in xrange(step):
                hdpgmm_un.sample(iter_step)
                hdpgmm_un.pickle(directory, 'hdpgmm_un_%d_run_%d-th_iter' % (run+1, (iter + 1) * iter_step + iter_start))

                hdpgmm_hl.sample(iter_step)
                hdpgmm_hl.pickle(directory, 'hdpgmm_hl_%d_run_%d-th_iter' % (run+1, (iter + 1) * iter_step + iter_start))

                print '%d-th iteration finished.' % ((iter + 1) * iter_step + iter_start)
                # compute log-likelihood
                weights_hl, dists_hl = dict2mix(hdpgmm_hl.params)
                weights_un, dists_un = dict2mix(hdpgmm_un.params)
                p0_all[:, iter] = np.array([all_loglike(d, weights_hl, dists_hl) for d in data_test_pca])
                p1_all[:, iter] = np.array([all_loglike(d, weights_un, dists_un) for d in data_test_pca])

            print '%d-th run completed...' % (run+1)
            p0 = np.sum(p0_all, axis=1)
            p1 = np.sum(p1_all, axis=1)
            pred = p0 < p1
            y = pH[test] <= threshold
            tmp_accuracy[run] = 1.*np.sum(pred == y)/len(y)
            tmp_tpr[run] = 1.*np.sum(pred & y)/np.sum(y)
            tmp_tnr[run] = 1.*np.sum(~pred & ~y)/(len(y) - np.sum(y))
            print 'tnr in the %dth run is ' % (run+1), tmp_tnr[run]
            print 'tpr in the %dth run is ' % (run+1), tmp_tpr[run]
            CV_idx[run] = {}
            CV_idx[run]['train'] = train
            CV_idx[run]['test'] = test
            run += 1

        pickle.dump(CV_idx, open(directory + 'CV_idx', 'wb'))
        np.save(directory + 'tpr', tmp_tpr)
        np.save(directory + 'tnr', tmp_tnr)
        print 'segment length is %ds' % (segLen / 4), 'q is %d' % q
        print 'true positive rate is ', np.mean(tmp_tpr), 'std is ', np.std(tmp_tpr)
        print 'true negative rate is ', np.mean(tmp_tnr), 'std is ', np.std(tmp_tnr)
        print 'wra is ', np.mean(tmp_tpr) + np.mean(tmp_tnr) - 1
