import numpy as np
import pickle
from model_loglikelihood import dict2mix, all_loglike
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt


def mix_multivariate_normal(weights, dists, n):
    counts = np.random.multinomial(n, weights)
    corpus = np.empty((0, len(dists[0].mean)))
    for i, c in enumerate(counts):
        tmp = np.random.multivariate_normal(dists[i].mean, dists[i].cov, c)
        corpus = np.concatenate((corpus, tmp), axis=0)
    return corpus


segLens = [ 120]
idx = np.load('./index/idx_705.npy')
feat_idx = np.arange(6)
pH = np.load('pH.npy')
pH = pH[idx]
threshold = 7.05
unhealthy = np.where(pH <= threshold)[0]
healthy = np.where(pH > threshold)[0]
label = pH <= threshold                  # 1 for unhealthy, 0 for healthy
q = 3
plot = False

for segLen in segLens:
    for run in xrange(2):
        feats = np.load('./features/feats_time_freq_%d.npy' % segLen)
        print 'segment length is', segLen, 'samples,', 'dimension is ', q
        data = feats[idx, :, :]  # use certain recordings according to idx
        # data = data[:, :, feat_idx]

        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        pca = PCA(n_components=q)
        shape = data.shape
        data_reshape = np.reshape(data, (shape[0] * shape[1], shape[2]))
        data_reshape_scaled = scaler.fit_transform(data_reshape)
        data_pca_reshape = pca.fit_transform(data_reshape_scaled)
        data_pca = np.reshape(data_pca_reshape, (shape[0], shape[1], q))

        directory = './results/2_model/no_CV_average_hyper_param_pca_scaled/time_freq_dim%d_%ds_feats_%dst_run/' \
                    % (q, segLen/4, (run+1))
        iter_start = 60
        iter_max = 80
        step_size = 10
        tnr = np.array([])
        tpr = np.array([])
        loglike_un = np.empty(0)
        loglike_hl = np.empty(0)
        for iter in xrange(iter_start, iter_max+1, step_size):
            hdpgmm_un = pickle.load(open(directory + 'hdpgmm_un_%d-th_iter' % iter))
            hdpgmm_hl = pickle.load(open(directory + 'hdpgmm_hl_%d-th_iter' % iter))
            # transform class object to dictionary
            weights_hl, dists_hl = dict2mix(hdpgmm_hl.params)
            weights_un, dists_un = dict2mix(hdpgmm_un.params)

            loglike_un = np.append(loglike_un, hdpgmm_un.log_likelihoods)
            loglike_hl = np.append(loglike_hl, hdpgmm_hl.log_likelihoods)
            if plot:
                # plot log-likelihood against number of iterations
                plt.figure()
                x = np.arange(1, iter + 1)
                plt.plot(x, loglike_un, 'r')
                plt.xlim(1, iter+1)
                plt.xlabel('iteration')
                plt.ylabel('log-likelihood')
                plt.show()
                plt.figure()
                plt.plot(x, loglike_hl, 'b')
                plt.xlim(1, iter+1)
                plt.xlabel('iteration')
                plt.ylabel('log-likelihood')
                plt.show()
            '''
            # generate synthetic data
            corpus_hl = mix_multivariate_normal(weights_hl, dists_hl, num)
            corpus_un = mix_multivariate_normal(weights_un, dists_un, num)
            corpus = np.concatenate((corpus_hl, corpus_un), axis=0)

            print 'number of clusters in healthy mixture is ', len(weights_hl)
            print 'number of clusters in unhealthy mixture is ', len(weights_un)

            p0 = np.array([mixture_logpdf(data, weights_hl, dists_hl) for data in corpus])
            p1 = np.array([mixture_logpdf(data, weights_un, dists_un) for data in corpus])
            '''
            p0 = np.array([all_loglike(d, weights_hl, dists_hl) for d in data_pca])
            p1 = np.array([all_loglike(d, weights_un, dists_un) for d in data_pca])
            pred = p0 < p1
            y = label
            tpr = np.append(tpr, 1.*np.sum(pred & y)/np.sum(y))
            tnr = np.append(tnr, 1.*np.sum(~pred & ~y)/(len(y) - np.sum(y)))

        print 'in the %drd run, segment length %ds, dimension %d' % ((run+3), segLen/4, q)
        print 'all tpr are ', tpr
        print 'all tnr are ', tnr
        print 'true positive rate is ', np.mean(tpr), 'std is ', np.std(tpr)
        print 'true negative rate is ', np.mean(tnr), 'std is ', np.std(tnr)
        print 'wra is ', np.mean(tpr)+np.mean(tnr)-1
