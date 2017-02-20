import numpy as np
import pickle
from scipy import signal
from sklearn import preprocessing


# compute short term variability
# two types: 1) standard STV = 1/N*|s(N)-s(1)|
#            2) STV-HAA = IQR(arctan(s(i+1)/s(i))), i=1,...,N
def stv(s, type):
    M = np.int(np.ceil(len(s)/60.))  # compute the number of minutes
    # first smooth FHR to get epoch-to-epoch variation, where each epoch contains 10 samples
    n_epoch = np.int(np.ceil(len(s)/10.))
    ee = np.array([np.mean(s[10*n:min(10*(n+1), len(s)-1)]) for n in xrange(n_epoch)])

    if type == 1:
        res = 0.
        for m in xrange(M):
            res += np.mean(np.abs(ee[6*m+1:min(6*(m+1)+1, len(ee))] - ee[6*m:min(6*(m+1), len(ee)-1)]))
        return res/M
    elif type == 2:
        res = 0.
        for m in xrange(M):
            q75, q25 = np.percentile(np.arctan(ee[m*6+1:min((m+1)*6, len(ee))]/ee[m*6:min((m+1)*6, len(ee))-1]), [75, 25])
            res += (q75-q25)
        return res/M


# compute long term variability
# two types: 1) delta value = 1/M*sum(max(s(i))-min(s(i)))
#            2) LTI = IQR(sqrt(T(i)^2 + T(i+1)^2))
def ltv(s, type):
    M = np.int_(np.ceil(len(s)/60.))  # compute the number of minutes
    if type == 1:
        res = 0.
        for m in xrange(M):
            res += max(s[m*60:(m+1)*60-1]) - min(s[m*60:(m+1)*60-1])
        return res/M
    elif type == 2:
        q75, q25 = np.percentile(np.sqrt(np.power(s[0:-1], 2) + np.power(s[1:], 2)), [75, 25])
        return q75-q25


# compute the auto-correlation of sequence x with lag-n
def autocorr(x, lag):
    corr = np.correlate(x, x, 'full')
    return corr[len(x)-1-lag]


# compute the Poincare plot descriptors: SD1, SD2, CCM
# Here I made approximation when calculating CCM
def poincare(s, m):
    sd1 = np.sqrt(2.)/2.*np.std(s[1:] - s[0:-1])
    # sd2 = np.sqrt(2*np.var(s)-1./2*np.var(s[1:] - s[0:-1]))
    sd2 = np.sqrt(autocorr(s, 0) + autocorr(s, 1) - 2*np.mean(s)**2)
    if sd1 == 0:
        ccm = 0
    else:
        ccm = (autocorr(s, m-2) - 2*autocorr(s, m-1) + 2*autocorr(s, m+1) - autocorr(s, m+2))/(2*np.pi*sd1*sd2*(len(s)-2))
    return sd1, sd2, ccm


# compute power in each frequency band
# VLF: 0-0.03Hz, LF: 0.03-0.15Hz, MF: 0.15-0.5Hz, HF:0.5-1Hz
def bandpower(s):
    f, p = signal.periodogram(s, fs=4., nfft=max(2*np.int_(fs/.03), len(s)))   # psd of T
    vlf = [0., .03]*2
    lf = [.03, .15]*2
    mf = [.15, .5]*2
    hf = [.5, 1.]*2
    vlf_idx = np.floor(np.multiply(vlf, len(f)))
    lf_idx = np.floor(np.multiply(lf, len(f)))
    mf_idx = np.floor(np.multiply(mf, len(f)))
    hf_idx = np.floor(np.multiply(hf, len(f)))
    p_vlf = np.trapz(p[vlf_idx[0]:vlf_idx[1]], f[vlf_idx[0]:vlf_idx[1]])
    p_lf = np.trapz(p[lf_idx[0]:lf_idx[1]], f[lf_idx[0]:lf_idx[1]])
    p_mf = np.trapz(p[mf_idx[0]:mf_idx[1]], f[mf_idx[0]:mf_idx[1]])
    p_hf = np.trapz(p[hf_idx[0]:hf_idx[1]], f[hf_idx[0]:hf_idx[1]])
    if p_mf+p_hf == 0.:
        ratio = 1.
    else:
        ratio = p_lf/(p_mf+p_hf)
    return p_vlf, p_lf, p_mf, p_hf, ratio


def UABandpower(s):
    f, p = signal.periodogram(s, fs=4., nfft=max(2*np.int_(fs/.03), len(s)))   # psd of T
    lf = [0., .5]
    mf = [.5, 1.]
    hf = [1., 2.]
    lf_idx = np.floor(np.multiply(lf, len(f)))
    mf_idx = np.floor(np.multiply(mf, len(f)))
    hf_idx = np.floor(np.multiply(hf, len(f)))
    p_lf = np.trapz(p[lf_idx[0]:lf_idx[1]], f[lf_idx[0]:lf_idx[1]])
    p_mf = np.trapz(p[mf_idx[0]:mf_idx[1]], f[mf_idx[0]:mf_idx[1]])
    p_hf = np.trapz(p[hf_idx[0]:hf_idx[1]], f[hf_idx[0]:hf_idx[1]])
    return p_lf, p_mf, p_hf


'''
# load .mat file and save to .npy file
alldata = sio.loadmat('all_eps.mat')
outcome = sio.loadmat('outcome_metrics.mat')
data = alldata['all_eps']
pH = outcome['outcome_metrics'][0][0].T[0]
fhr = {}
bline = {}
ind = 0
for seq in data[:, 0]:
    # skip if empty
    fhr[ind] = seq['fhrMedFilt'][0][0].T[0]
    bline[ind] = seq['blineFhr'][0][0].T[0]
    ind += 1
print 'Number of sub FHR segment is', len(fhr)

pickle.dump(fhr, open('fhr.npy', 'wb'))
pickle.dump(bline, open('bline.npy', 'wb'))
'''
fs = 4                  # 4Hz sampling rate
duration = 35*60*fs     # analyze the last 30-min data
window = 5*60*fs
segLens = [40, 80, 120, 160]            # 40 samples/10 seconds per segment
fhr = pickle.load(open('../../data/CTU-UHB/fhr.pickle', 'rb'))
# fhr = np.load('fhr.npy')
ua = pickle.load(open('../../data/CTU-UHB/ua.pickle', 'rb'))
bline = pickle.load(open('../../data/CTU-UHB/bline.pickle', 'rb'))
# feats_time_freq_whole_seg = {}
scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
for segLen in segLens:
    numSeg = (duration-window)/segLen
    feat_vector_fhr = np.empty([len(fhr), numSeg, 14])
    feat_vector_ua = np.empty([len(ua), numSeg, 4])
    # feats_time_freq_whole_seg[segLen] = {}
    for idx in fhr:
        if len(fhr[idx]) < duration:
            continue
        # y = fhr[idx]                            # ignore the duration, use all the data
        # y_debline = y - bline[idx]
        y = fhr[idx][-duration:]                # analyze the last segment of data
        y_ua_raw = ua[idx][-duration:]
        y_ua = scaler.fit_transform(y_ua_raw.reshape(-1, 1))
        y_ua = y_ua.reshape(-1)
        y_debline = y - bline[idx][-duration:]  # get fhr after removing baseline
        rr = np.reciprocal(y/60.)                # get RR interval

        # numSeg = len(y)/segLen
        # feat_vector = np.empty([numSeg, 14])

        for t in xrange(numSeg):
            s = y[segLen*(t+1):segLen*(t+1)+window]
            r = rr[segLen*(t+1):segLen*(t+1)+window]
            s_dbl = y_debline[segLen*(t+1):segLen*(t+1)+window]
            s_ua = y_ua[segLen*(t+1):segLen*(t+1)+window]
            # s_debline = y_debline[segLen*t:segLen*(t+1)]
            # FHR features
            mean = np.mean(s)
            std = np.std(s)
            stv_std = stv(s, 1)
            stv_sti = stv(r, 2)
            ltv_delta = ltv(s, 1)
            ltv_lti = ltv(r, 2)
            sd1, sd2, ccm1 = poincare(r, 1)
            power_vlf, power_lf, power_mf, power_hf, ratio = bandpower(s_dbl)
            feat_vector_fhr[idx, t, :] = [mean, std, stv_std, stv_sti, ltv_delta, ltv_lti, sd1, sd2, ccm1,
                                          power_vlf, power_lf, power_mf, power_hf, ratio]
            # UA features
            mean_ua = np.mean(s_ua)
            std_ua = np.std(s_ua)
            power_lf_ua, power_mf_ua, power_hf_ua = UABandpower(s_ua-mean_ua)
            feat_vector_ua[idx, t, :] = [mean_ua, std_ua, power_lf_ua, power_mf_ua]
        # feats_time_freq_whole_seg[segLen][idx] = feat_vector
        print '%d-th recording extraction complete' % idx
    np.save('./features/5min_featsFHR_time_freq_%d' % segLen, feat_vector_fhr)
    np.save('./features/5min_featsUA_time_freq_%d' % segLen, feat_vector_ua)
    print '%d samples finished.' % segLen

# np.save('./features/healhty_feats_time_freq_%d' % segLen, feat_vector)
# pickle.dump(feats_time_freq_whole_seg, open('./features/feats_time_freq_whole_seg.pickle', 'wb'))

'''
feats = np.load('feat_vector.npy')
outcome = sio.loadmat('outcome_metrics.mat')
pH = outcome['outcome_metrics'][0][0].T[0]
diag = pH < 7.15    # True represents unhealthy

# use SVR to predict pH value
target = np.multiply((pH - np.min(pH)), 10./(np.max(pH)-np.min(pH)))
feat_vector = feats[:, 2:]
kf = cross_validation.KFold(len(feat_vector), n_folds=5)
Cs = [1., 10., 100.]
epislons = [.00001, .0001, .001, .01, .1]
gammas = [.1, 1., 10.]
scores = np.empty(len(epislons))
for idx, epislon in enumerate(epislons):
    score = 0.
    for train_idx, test_idx in kf:
        X_train, X_test = feat_vector[train_idx], feat_vector[test_idx]
        y_train, y_test = target[train_idx], target[test_idx]
        svr = SVR(C=100., epsilon=epislon)
        svr.fit(X_train, y_train)
        prd = svr.predict(X_test)
        tmp = r2_score(y_test, prd)
        print(tmp)
        score += tmp
    scores[idx] = score/len(kf)
print scores
'''
