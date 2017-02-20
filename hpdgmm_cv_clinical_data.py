import numpy as np
from model_loglikelihood import all_loglike, dict2mix
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd


idx = np.load('./index/idx_705.npy')
pH = np.load('pH.npy')
clinical = pd.read_csv('clinical.df').iloc[:, 1:]
pH = pH[idx]
clinical = clinical.iloc[idx, :]
threshold = 7.05
unhealthy = np.where(pH <= threshold)[0]
healthy = np.where(pH > threshold)[0]
label = (pH <= threshold).astype(int)                  # 1 for unhealthy, 0 for healthy
n_fold = 5

segLens = [40, 80]
qs = [2, 3]                           # dimension after PCA
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

        folder = 'time_freq_dim%d_%ds_FHR_2nd_run' % (q, segLen / 4)
        directory = './results/2_model/CV_hyper_param_pca_scaled_7.05/%s/' % folder

        tmp_accuracy = np.zeros(n_fold)
        tmp_tnr = np.zeros(n_fold)
        tmp_tpr = np.zeros(n_fold)
        run = 0
        CV_idx = pickle.load(open(directory + 'CV_idx', 'rb'))

        for run in CV_idx:
            train = CV_idx[run]['train']
            test = CV_idx[run]['test']

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

            p0_all = np.empty((len(test), step))
            p1_all = np.empty((len(test), step))
            # average sampling results from iter_start to iter_stop for every iter_step iterations
            for iter in xrange(step):
                hdpgmm_un = pickle.load(open(directory + 'hdpgmm_un_%d_run_%d-th_iter' % (run+1, (iter + 1) * iter_step + iter_start)))
                hdpgmm_hl = pickle.load(open(directory + 'hdpgmm_hl_%d_run_%d-th_iter' % (run+1, (iter + 1) * iter_step + iter_start)))

                # compute log-likelihood
                weights_hl, dists_hl = dict2mix(hdpgmm_hl.params)
                weights_un, dists_un = dict2mix(hdpgmm_un.params)
                p0_all[:, iter] = np.array([all_loglike(d, weights_hl, dists_hl) for d in data_test_pca])
                p1_all[:, iter] = np.array([all_loglike(d, weights_un, dists_un) for d in data_test_pca])

            print '%d-th run completed...' % (run+1)
            p0 = np.sum(p0_all, axis=1)
            p1 = np.sum(p1_all, axis=1)

            numSeg = 30 * 60 * 4 / segLen
            confidence = np.exp(p0 / numSeg) / (np.exp(p0 / numSeg) + np.exp(p1 / numSeg))

            p0h = 23/44.; p1h = 21/44.
            p0u = 39/44.; p1u = 5/44.

            priors_h = [p0h/(p0h+p0u) if clinical.loc[i, 'Parity'] < 1 else p1h/(p1h+p1u) for i in idx[test]]
            priors_u = [p0u/(p0h+p0u) if clinical.loc[i, 'Parity'] < 1 else p1u/(p1h+p1u) for i in idx[test]]
            p0 = np.multiply(confidence, priors_h)
            p1 = np.multiply(1 - confidence, priors_u)

            pred = p0 < p1
            y = pH[test] <= threshold
            tmp_accuracy[run] = 1.*np.sum(pred == y)/len(y)
            tmp_tpr[run] = 1.*np.sum(pred & y)/np.sum(y)
            tmp_tnr[run] = 1.*np.sum(~pred & ~y)/(len(y) - np.sum(y))
            # print 'tnr in the %dth run is ' % (run+1), tmp_tnr[run]
            # print 'tpr in the %dth run is ' % (run+1), tmp_tpr[run]

            run += 1

        # pickle.dump(CV_idx, open(directory + 'CV_idx', 'wb'))
        # np.save(directory + 'tpr', tmp_tpr)
        # np.save(directory + 'tnr', tmp_tnr)
        # print 'segment length is %ds' % (segLen / 4), 'q is %d' % q
        print 'true positive rate is ', np.mean(tmp_tpr), 'std is ', np.std(tmp_tpr)
        print 'true negative rate is ', np.mean(tmp_tnr), 'std is ', np.std(tmp_tnr)
        print 'wra is ', np.mean(tmp_tpr) + np.mean(tmp_tnr) - 1
