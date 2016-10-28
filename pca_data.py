import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import preprocessing


segLen = 240
q = 2

data = np.load('./features/feats_time_freq_%d.npy' % segLen)
scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
pca = PCA(n_components=q)
shape = data.shape
data_reshape = np.reshape(data, (shape[0] * shape[1], shape[2]))
data_reshape_scaled = scalar.fit_transform(data_reshape)
data_reshape_scaled_pca = pca.fit_transform(data_reshape_scaled)
data_scaled_pca = np.reshape(data_reshape_scaled_pca, (shape[0], shape[1], q))
print ''
sio.savemat('./features/matlab/feats_time_freq_%ds' % (segLen/4), mdict={'data':data_scaled_pca})
