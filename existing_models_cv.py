import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold


idx = np.load('./index/idx_705.npy')
pH = np.load('pH.npy')
pH = pH[idx]
threshold = 7.05
unhealthy = np.where(pH <= threshold)[0]
healthy = np.where(pH > threshold)[0]
label = pH <= threshold                  # 1 for unhealthy, 0 for healthy
skf = StratifiedKFold(label, 10)

feats = np.load('./features/feats_time_freq_7200.npy')
data = feats[idx, :, :]
data = np.reshape(data, (data.shape[0], data.shape[2]))

min_max_scaler = preprocessing.MinMaxScaler()
# data_scaled = min_max_scaler.fit_transform(data)
data_scaled = preprocessing.scale(data)

tmp_tnr = np.zeros(len(skf))
tmp_tpr = np.zeros(len(skf))
run = 0
CV_idx = {}

for train, test in skf:
    X_train, X_test = data_scaled[train], data_scaled[test]
    y_train, y_test = label[train], label[test]

    # pca = PCA(n_components=q)
    # X_train_pca = pca.fit_transform(X_train)

    clf = SVC(C=3., kernel='rbf', gamma='auto')
    clf.fit(X_train, y_train)

    # X_test_pca = pca.transform(X_test)
    pred = clf.predict(X_test)
    tmp_tpr[run] = 1. * np.sum(pred & y_test) / np.sum(y_test)
    tmp_tnr[run] = 1. * np.sum(~pred & ~y_test) / (len(y_test) - np.sum(y_test))

    print 'tnr in the %dth run is ' % (run + 1), tmp_tnr[run], 'tpr in the %dth run is ' % (run + 1), tmp_tpr[run]
    CV_idx[run] = {}
    CV_idx[run]['train'] = train
    CV_idx[run]['test'] = test
    run += 1

print 'true positive rate is ', np.mean(tmp_tpr), 'std is ', np.std(tmp_tpr)
print 'true negative rate is ', np.mean(tmp_tnr), 'std is ', np.std(tmp_tnr)
print 'wra is ', np.mean(tmp_tpr) + np.mean(tmp_tnr) - 1
