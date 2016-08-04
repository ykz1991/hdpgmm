import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.cross_validation import StratifiedKFold
from model_loglikelihood import dict2mix, mixture_logpdf, all_loglike
from hdpgmm_class_v2 import GibbsSampler
np.seterr(divide='ignore')

# selecting model parameters: segment length
segLens = [100, 220]
idx = np.load('idx_ctu.npy')
pH = np.load('pH.npy')
pH = pH[idx]
threshold = 7.15
unhealthy = np.where(pH < threshold)[0]
healthy = np.where(pH >= threshold)[0]
label = pH < threshold                  # 1 for unhealthy, 0 for healthy
skf = StratifiedKFold(label, 10)
for segLen in segLens:
    feats = np.load('./features/feats_time_freq_%d.npy' % segLen)
    print 'segment length is ', segLen, 'samples'
    data = feats[idx, :, :]                 # use certain recordings according to idx
    # data = data[:, :, [0, 1, 2, 3, 6, 7]]   # ARX coefficients, std, mean of rr interval
    # data = data[:, :, [0, 3, 5, 6, 7, 8]]   # mean, sti, lti, poincare
    unhealthy_data = data[unhealthy]
    healthy_data = data[healthy]

    folder = 'time_freq_%ds' % (segLen/4)
    directory = './results/2_model/CV_hyper_param/%s/' % folder
    if not os.path.isdir(directory):
        os.makedirs(directory)

    tmp_accuracy = np.zeros(len(skf))
    tmp_tnr = np.zeros(len(skf))
    tmp_tpr = np.zeros(len(skf))
    run = 0
    CV_idx = {}
    iter_start = 100
    iter_stop = 150
    iter_step = 10
    step = (iter_stop - iter_start) / iter_step
    for train, test in skf:
        # train two HDPGMM models
        # initialization
        hdpgmm_un = GibbsSampler(snapshot_interval=20)
        hdpgmm_hl = GibbsSampler(snapshot_interval=20)
        hdpgmm_un._initialize(data[train[pH[train] < threshold]])
        hdpgmm_hl._initialize(data[train[pH[train] >= threshold]])

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
            p0_all[:, iter] = np.array([all_loglike(data[i], weights_hl, dists_hl) for i in test])
            p1_all[:, iter] = np.array([all_loglike(data[i], weights_un, dists_un) for i in test])

        print '%d-th run completed...' % (run+1)
        p0 = np.sum(p0_all, axis=1)
        p1 = np.sum(p1_all, axis=1)
        pred = (p0 < p1).astype(int)
        y = (pH[test] < threshold).astype(int)
        tmp_accuracy[run] = 1.*np.sum(pred == y)/len(y)
        tmp_tpr[run] = 1.*np.sum(pred.astype(bool) & y.astype(bool))/np.sum(y)
        tmp_tnr[run] = 1.*np.sum(~pred.astype(bool) & ~y.astype(bool))/(len(y) - np.sum(y))
        CV_idx[run] = {}
        CV_idx[run]['train'] = train
        CV_idx[run]['test'] = test
        run += 1

    pickle.dump(CV_idx, open(directory + 'CV_idx', 'wb'))
    np.save(directory + 'tpr', tmp_tpr)
    np.save(directory + 'tnr', tmp_tnr)
    print 'segment length is %ds' % (segLen / 4)
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
