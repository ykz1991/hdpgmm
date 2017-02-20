import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from model_loglikelihood import dict2mix, all_loglike
from hdpgmm_class_v2 import GibbsSampler
from sklearn.decomposition import PCA
from sklearn import preprocessing

np.seterr(divide='ignore')

# This version investigates jointly processing FHR and UA signals (comparison with v3, v4)
# FHR and UA are combined before PCA

idx = np.load('./index/idx_710_ua_35min.npy')
pH = np.load('pH.npy')
pH = pH[idx]
threshold = 7.1
unhealthy = np.where(pH <= threshold)[0]
healthy = np.where(pH > threshold)[0]
label = (pH <= threshold).astype(int)                  # 1 for unhealthy, 0 for healthy
n_fold = 5
skf = StratifiedKFold(n_splits=n_fold)

segLens = [40, 80, 120]
qs = [5, 6]                           # dimension after PCA
# Number of iteration in sampling
iter_start = 80
iter_stop = 100
iter_step = 10
step = (iter_stop - iter_start) / iter_step
for q in qs:
    for segLen in segLens:
        feats_fhr = np.load('./features/5min_featsFHR_time_freq_%d.npy' % segLen)
        feats_ua = np.load('./features/5min_featsUA_time_freq_%d.npy' % segLen)
        feats = np.concatenate((feats_fhr, feats_ua), axis=2)
        print 'segment length is ', segLen, 'samples'
        data = feats[idx, :, :]                 # use certain recordings according to idx

        folder = 'time_freq_dim%d_%ds_FHR_UA_beforePCA' % (q, segLen/4)
        directory = './results/2_model/CV_hyper_param_pca_scaled_7.1_UA_movingWin/%s/' % folder
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
            hdpgmm_un = GibbsSampler(snapshot_interval=50)
            hdpgmm_hl = GibbsSampler(snapshot_interval=50)
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
'''
alpha = 1.
gamma = 1.
folder = 'time_freq_feats_alpha_%d_gamma_%d' % (alpha, gamma)
CV_idx = pickle.load(open('./results/2_model/CV/%s/CV_idx' % folder, 'rb'))
# plt.figure()
for run in CV_idx:
    hdpgmm_un = pickle.load(open('./results/2_model/CV/%s/hdpgmm_un_%i_run' % (folder, run), 'rb'))
    hdpgmm_hl = pickle.load(open('./results/2_model/CV/%s/hdpgmm_hl_%i_run' % (folder, run), 'rb'))

    log_likelihood_un = hdpgmm_un.log_likelihoods
    log_likelihood_hl = hdpgmm_hl.log_likelihoods
    x = np.arange(1, 16)
    plt.plot(x, log_likelihood_un, 'b', x, log_likelihood_hl, 'r')

    test = CV_idx[run]['test']
    weights_hl, dists_hl = dict2mix(hdpgmm_hl.params)
    weights_un, dists_un = dict2mix(hdpgmm_un.params)
    p0 = np.array([all_loglike(data[i], weights_hl, dists_hl) for i in test])
    p1 = np.array([all_loglike(data[i], weights_un, dists_un) for i in test])
    pred = (p0 < p1).astype(int)
    y = (pH[test] < threshold).astype(int)
    tmp_tpr[run] = 1.*np.sum(pred.astype(bool) & y.astype(bool))/np.sum(y)
    tmp_tnr[run] = 1.*np.sum(~pred.astype(bool) & ~y.astype(bool))/(len(y) - np.sum(y))

plt.xlim(1, 15)
plt.xlabel('iteration')
plt.ylabel('log-likelihood')
plt.show()

tpr = np.mean(tmp_tpr)
tnr = np.mean(tmp_tnr)
accuracy = np.mean(tmp_accuracy)
np.save('./results/2_model/CV/%s/tpr' % folder, tmp_tpr)
np.save('./results/2_model/CV/%s/tnr' % folder, tmp_tnr)
print 'true positive rate is ', tpr
print 'true negative rate is ', tnr
print 'accuracy is ', accuracy
'''
