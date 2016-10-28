import numpy as np
from model_loglikelihood import all_loglike, dict2mix
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn import preprocessing


# specify parameters
segLen = 80
q = 3

# load data
feats = pickle.load(open('./features/feats_time_freq_whole_seg.pickle', 'rb'))
pH = np.load('pH.npy')
idx = np.load('./index/idx_705.npy')
# idx = np.where(pH <= 7.05)[0]
ind = 193
data = feats[segLen][ind]

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

# shape = data.shape
# data_reshape = np.reshape(data, (shape[0]*shape[1], shape[2]))
data_scaled = scaler.transform(data)
data_pca = pca.transform(data_scaled)

print data_pca.shape
# load hpdgmm class instance
directory = './results/2_model/no_CV_average_hyper_param_pca_scaled/time_freq_dim%d_%ds_feats_2st_run/' % (q, segLen/4)

length = data_pca.shape[0] - 30*60/segLen + 1
prob_hl = np.zeros([length, 3])
prob_un = np.zeros([length, 3])

num = 0
for iter in xrange(60, 81, 10):
    hdpgmm_un = pickle.load(open(directory + 'hdpgmm_un_%d-th_iter' % iter))
    hdpgmm_hl = pickle.load(open(directory + 'hdpgmm_hl_%d-th_iter' % iter))
    weights_hl, dists_hl = dict2mix(hdpgmm_hl.params)
    weights_un, dists_un = dict2mix(hdpgmm_un.params)
    prob_hl[:, num] = np.array([all_loglike(data_pca[i:i+30*60/segLen], weights_hl, dists_hl) for i in xrange(length)])
    prob_un[:, num] = np.array([all_loglike(data_pca[i:i+30*60/segLen], weights_un, dists_un) for i in xrange(length)])
    num += 1

p0 = np.mean(prob_hl, axis=1)
p1 = np.mean(prob_un, axis=1)

numSeg = 30*60*4/segLen
confidence = 100 * np.exp(p0/numSeg) / (np.exp(p0/numSeg) + np.exp(p1/numSeg))

plt.plot(confidence)
plt.title('pH value is %.2f' % pH[ind])
plt.show()
