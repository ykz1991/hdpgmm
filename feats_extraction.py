import numpy as np
import pickle
from scipy import signal


# compute short term variability
# two types: 1) standard STV = 1/N*|s(N)-s(1)|
#            2) STV-HAA = IQR(arctan(s(i+1)/s(i))), i=1,...,N
def stv(s, type):
    M = np.int_(np.ceil(len(s)/60.))  # compute the number of minutes
    if type == 1:
        res = 0.
        for m in xrange(M):
            res += np.mean(np.abs(s[60*m+1:min(60*(m+1)+1, len(s))] - s[60*m:min(60*(m+1), len(s)-1)]))
        return res/M
    elif type == 2:
        res = 0.
        for m in xrange(M):
            q75, q25 = np.percentile(np.arctan(s[m*60+1:min((m+1)*60, len(s))]/s[m*60:min((m+1)*60, len(s))-1]), [75, 25])
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
    vlf = [0., .03]
    lf = [.03, .15]
    mf = [.15, .5]
    hf = [.5, 1.]
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
duration = 30*60*fs     # analyze the last 30-min data
segLen = 100             # 40 samples/10 seconds per segment
numSeg = duration/segLen
fhr = pickle.load(open('fhr.npy', 'rb'))
bline = pickle.load(open('bline.npy', 'rb'))
feat_vector = np.empty([len(fhr), numSeg, 14])
for idx in fhr:
    y = fhr[idx][-duration:]                # analyze the last segment of data
    y_debline = y - bline[idx][-duration:]  # get fhr after removing baseline
    t = np.reciprocal(y/60.)                # get RR interval
    for t in xrange(numSeg):
        s = y[segLen*t:segLen*(t+1)]
        s_debline = y_debline[segLen*t:segLen*(t+1)]
        mean = np.mean(s)
        var = np.var(s)
        stv_std = stv(s, 1)
        stv_haa = stv(s, 2)
        ltv_delta = ltv(s, 1)
        ltv_lti = ltv(s, 2)
        sd1, sd2, ccm1 = poincare(s, 1)
        power_vlf, power_lf, power_mf, power_hf, ratio = bandpower(s)
        feat_vector[idx, t] = [mean, var, stv_std, stv_haa, ltv_delta, ltv_lti, sd1, sd2, ccm1,
                               power_vlf, power_lf, power_mf, power_hf, ratio]
    print '%d-th feature extraction complete' % idx
np.save('./features/feats_time_freq_%d' % segLen, feat_vector)


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