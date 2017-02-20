from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np

idx = np.load('idx_ctu.npy')
pH = np.load('pH.npy')
pH = pH[idx]
threshold = 7.15
unhealthy = np.where(pH < threshold)[0]
healthy = np.where(pH >= threshold)[0]
label = pH < threshold                  # 1 for unhealthy, 0 for healthy
q = 2

segLens = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
for segLen in segLens:
    feats = np.load('./features/feats_time_freq_%d.npy' % segLen)
    print 'segment length is ', segLen, 'samples'
    data = feats[idx, :, :]                 # use certain recordings according to idx
    unhealthy_data = data[unhealthy]
    healthy_data = data[healthy]

    # use PCA to reduce the dimension to 2
    shape = unhealthy_data.shape
    unhealthy_data_reshape = np.reshape(unhealthy_data, (shape[0]*shape[1], shape[2]))
    shape = healthy_data.shape
    healthy_data_reshape = np.reshape(healthy_data, (shape[0]*shape[1], shape[2]))
    pca_un = PCA(n_components=q)
    pca_hl = PCA(n_components=q)

    '''
    pca_hl.fit(healthy_data_reshape)
    pca_un.fit(unhealthy_data_reshape)
    print 'healthy var ratio ', pca_hl.explained_variance_ratio_
    print 'unhealthy var ratio ', pca_un.explained_variance_ratio_
    '''
    unhealthy_data_pca_reshape = pca_un.fit_transform(unhealthy_data_reshape)
    healthy_data_pca_reshape = pca_hl.fit_transform(healthy_data_reshape)
    unhealthy_data_pca = np.reshape(unhealthy_data_pca_reshape, (unhealthy_data.shape[0], unhealthy_data.shape[1], q))
    healthy_data_pca = np.reshape(healthy_data_pca_reshape, (healthy_data.shape[0], healthy_data.shape[1], q))

    directory = './pca_data/3dim/'
    np.save(directory + 'unhealthy_data_pca_%ds' % (segLen/4), unhealthy_data_pca)
    np.save(directory + 'healthy_data_pca_%ds' % (segLen/4), healthy_data_pca)
