import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import os
from sklearn.cross_validation import StratifiedKFold
from model_loglikelihood import dict2mix, mixture_logpdf, all_loglike
from hdpgmm_class import Gaussian, GibbsSampler
np.seterr(divide='ignore')

# selecting model and feature parameters: segment length, alpha, gamma
segLens = [20, 60, 100, 120]
alphas = [5.]
gammas = [10.]
idx = np.load('idx_ctu.npy')
for segLen in segLens:
    feats = np.load('./features/feats_time_freq_%d.npy' % segLen)
    print 'segment length is ', segLen, 'samples'
    pH = np.load('pH.npy')
    data = feats[idx, :, :]                 # use certain recordings according to idx
    pH = pH[idx]
    threshold = 7.15
    # data = data[:, :, [0, 1, 2, 3, 6, 7]]   # ARX coefficients, std, mean of rr interval
    data = data[:, :, [0, 3, 5, 6, 7, 8]]   # mean, sti, lti, poincare
    unhealthy = np.where(pH < threshold)[0]
    healthy = np.where(pH >= threshold)[0]
    label = pH < threshold                  # 1 for unhealthy, 0 for healthy
    unhealthy_data = data[unhealthy]
    healthy_data = data[healthy]
    skf = StratifiedKFold(label, 5)
    tmp_accuracy = np.zeros(len(skf))
    tmp_tnr = np.zeros(len(skf))
    tmp_tpr = np.zeros(len(skf))

    for alpha in alphas:
        for gamma in gammas:
            print 'alpha is', alpha, ', gamma is', gamma
            folder = 'time_%ds_6feats_alpha_%d_gamma_%d' % (segLen/4, alpha, gamma)
            directory = './results/2_model/CV/%s/' % folder
            if not os.path.isdir(directory):
                os.makedirs(directory)

            run = 0
            CV_idx = {}
            iteration = 20
            for train, test in skf:
                # train two HDPGMM models
                hdpgmm_un = GibbsSampler(snapshot_interval=5)
                hdpgmm_un._initialize(data[train[pH[train] < threshold]], alpha=alpha, gamma=gamma)
                hdpgmm_un.sample(iteration)
                hdpgmm_un.pickle(directory, 'hdpgmm_un_%i_run' % run)

                hdpgmm_hl = GibbsSampler(snapshot_interval=5)
                hdpgmm_hl._initialize(data[train[pH[train] >= threshold]], alpha=alpha, gamma=gamma)
                hdpgmm_hl.sample(iteration)
                hdpgmm_hl.pickle(directory, 'hdpgmm_hl_%i_run' % run)

                # compute log-likelihood
                weights_hl, dists_hl = dict2mix(hdpgmm_hl.params)
                weights_un, dists_un = dict2mix(hdpgmm_un.params)
                p0 = np.array([all_loglike(data[i], weights_hl, dists_hl) for i in test])
                p1 = np.array([all_loglike(data[i], weights_un, dists_un) for i in test])
                pred = (p0 < p1).astype(int)
                y = (pH[test] < threshold).astype(int)
                tmp_accuracy[run] = 1.*np.sum(pred == y)/len(y)
                tmp_tpr[run] = 1.*np.sum(pred.astype(bool) & y.astype(bool))/np.sum(y)
                tmp_tnr[run] = 1.*np.sum(~pred.astype(bool) & ~y.astype(bool))/(len(y) - np.sum(y))
                CV_idx[run] = {}
                CV_idx[run]['train'] = train
                CV_idx[run]['test'] = test
                run += 1

            pickle.dump(CV_idx, open('./results/2_model/CV/%s/CV_idx' % folder, 'wb'))
            np.save('./results/2_model/CV/%s/tpr' % folder, tmp_tpr)
            np.save('./results/2_model/CV/%s/tnr' % folder, tmp_tnr)
            tpr = np.mean(tmp_tpr)
            tnr = np.mean(tmp_tnr)
            accuracy = np.mean(tmp_accuracy)
            print 'true positive rate is ', tpr
            print 'true negative rate is ', tnr
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
    tmp_accuracy[run] = 1.*np.sum(pred == y)/len(y)
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
