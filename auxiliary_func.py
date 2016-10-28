import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

segLen = 80
idx = np.load('./index/idx_705.npy')
feats = np.load('./features/feats_time_freq_%d.npy' % segLen)
data = feats[idx, :, :]
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
pca = PCA()
shape = data.shape
data_reshape = np.reshape(data, (shape[0]*shape[1], shape[2]))
data_reshape_scaled = scaler.fit_transform(data_reshape)
data_pca_reshape = pca.fit_transform(data_reshape_scaled)

x = range(14)
y = pca.explained_variance_ratio_
y *= 100
xx = np.arange(14) + .5
yy = np.cumsum(y)


plt.bar(x, y, width=1., color='grey')
plt.plot(xx, yy, color='blue')
plt.ylim((0, 110))
plt.grid()
plt.xlabel('Principle component')
plt.ylabel('Explained variance ratio (%)')
plt.show()
