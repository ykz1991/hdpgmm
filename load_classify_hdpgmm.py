import numpy as np
import pickle
from model_loglikelihood import dict2mix, all_loglike
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn import preprocessing
from sklearn.decomposition import PCA

# specify parameters
segLen = 120
q = 2

# load data
# feats = np.load('./features/simulated/unhealthy_feats_time_freq_%d.npy' % segLen)
data = np.load('./features/simulated/feats_time_freq_%d.npy' % segLen)
pH = np.load('pH.npy')
idx = np.load('./index/idx_705.npy')
# idx = np.where(pH <= 7.05)[0]
# data = feats[idx]
pH = pH[idx]

# perform PCA
# idx = np.load('./index/idx_705.npy')
feats = np.load('./features/feats_time_freq_%d.npy' % segLen)
data_tmp = feats[idx, :, :]
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
pca = PCA(n_components=q)
shape = data_tmp.shape
data_reshape = np.reshape(data_tmp, (shape[0]*shape[1], shape[2]))
data_reshape_scaled = scaler.fit_transform(data_reshape)
pca.fit(data_reshape_scaled)

shape = data.shape
data_reshape = np.reshape(data, (shape[0]*shape[1], shape[2]))
data_reshape_scaled = scaler.transform(data_reshape)
data_pca_reshape = pca.transform(data_reshape_scaled)
data_pca = np.reshape(data_pca_reshape, (shape[0], shape[1], q))

# load hpdgmm class instance
directory = './results/2_model/no_CV_average_hyper_param_pca_scaled/time_freq_dim%d_%ds_feats_1st_run/' % (q, segLen/4)

p0_all = np.empty((data_pca.shape[0], 3))
p1_all = np.empty((data_pca.shape[0], 3))

num = 0
for iter in xrange(60, 81, 10):
    hdpgmm_un = pickle.load(open(directory + 'hdpgmm_un_%d-th_iter' % iter))
    hdpgmm_hl = pickle.load(open(directory + 'hdpgmm_hl_%d-th_iter' % iter))

    weights_hl, dists_hl = dict2mix(hdpgmm_hl.params)
    weights_un, dists_un = dict2mix(hdpgmm_un.params)
    p0_all[:, num] = np.array([all_loglike(d, weights_hl, dists_hl) for d in data_pca])
    p1_all[:, num] = np.array([all_loglike(d, weights_un, dists_un) for d in data_pca])

    num += 1

p0 = np.mean(p0_all, axis=1)
p1 = np.mean(p1_all, axis=1)

pred = p0 < p1
y = np.concatenate((np.zeros(100), np.ones(100))).astype(bool)
tpr = 1.*np.sum(pred & y)/np.sum(y)
tnr = 1.*np.sum(~pred & ~y)/(len(y) - np.sum(y))
accuracy = 1.*np.sum(pred == y)/len(y)
print 'tpr is ', tpr, 'tnr is ', tnr
print 'accuracy is ', accuracy

numSeg = 30*60*4/segLen
confidence = np.exp(p0/numSeg) / (np.exp(p0/numSeg) + np.exp(p1/numSeg))

# plt.scatter(confidence[0:44]*100, pH[0:44])
# plt.show()
