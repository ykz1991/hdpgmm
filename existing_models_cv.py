import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import cross_validation


idx = np.load('./index/idx_705.npy')
pH = np.load('pH.npy')
pH = pH[idx]
threshold = 7.05
unhealthy = np.where(pH <= threshold)[0]
healthy = np.where(pH > threshold)[0]
label = pH <= threshold                  # 1 for unhealthy, 0 for healthy
skf = cross_validation.StratifiedKFold(label, 5, shuffle=False)

feats = np.load('./features/feats_time_freq_7200.npy')
data = feats[idx, :, :]
data = np.reshape(data, (data.shape[0], data.shape[2]))

tmp_tnr, tmp_tpr = np.zeros(len(skf)), np.zeros(len(skf))
tmp_accuracy = np.zeros((len(skf)))
tmp_tpr_train, tmp_tnr_train = np.zeros(len(skf)), np.zeros(len(skf))
run = 0
CV_idx = {}

for train, test in skf:
    X_train, X_test = data[train], data[test]
    y_train, y_test = label[train], label[test]

    # pca = PCA(n_components=q)
    # X_train_pca = pca.fit_transform(X_train)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = min_max_scaler.fit_transform(X_train)

    clf = SVC(C=3., kernel='rbf', gamma='auto')
    clf.fit(X_train_scaled, y_train)

    # X_test_pca = pca.transform(X_test)
    X_test_scaled = min_max_scaler.transform(X_test)
    pred = clf.predict(X_test_scaled)
    tmp_tpr[run] = 1. * np.sum(pred & y_test) / np.sum(y_test)
    tmp_tnr[run] = 1. * np.sum(~pred & ~y_test) / (len(y_test) - np.sum(y_test))

    pred_train = clf.predict(X_train)
    tmp_accuracy[run] = clf.score(X_test_scaled, y_test)
    tmp_tpr_train[run] = 1. * np.sum(pred_train & y_train) / np.sum(y_train)
    tmp_tnr_train[run] = 1. * np.sum(~pred_train & ~y_train) / (len(y_train) - np.sum(y_train))

    # print 'training tnr in the %dth run is ' % (run + 1), tmp_tnr_train[run], \
    #    'training tpr in the %dth run is ' % (run + 1), tmp_tpr_train[run]
    print 'tnr in the %dth run is ' % (run + 1), tmp_tnr[run], 'tpr in the %dth run is ' % (run + 1), tmp_tpr[run],\
        'accuracy is ', tmp_accuracy[run]
    CV_idx[run] = {}
    CV_idx[run]['train'] = train
    CV_idx[run]['test'] = test
    run += 1

print 'true positive rate is ', np.mean(tmp_tpr), 'std is ', np.std(tmp_tpr)
print 'true negative rate is ', np.mean(tmp_tnr), 'std is ', np.std(tmp_tnr)
print 'accuracy is ', np.mean(tmp_accuracy), 'std is ', np.std(tmp_accuracy)
print 'wra is ', np.mean(tmp_tpr) + np.mean(tmp_tnr) - 1
