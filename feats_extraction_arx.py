import numpy as np
import pickle


def stv(t):
    q75, q25 = np.percentile(np.arctan(t[0:-1]/t[1:]), [75, 25])
    return q75-q25


FHR = pickle.load(open('fhr.npy', 'rb'))
Bline = pickle.load(open('bline.npy', 'rb'))

SegLen = 40.
order = 1
# use 7 features, the first four are ARX model coefficents, the following two are variability (STV and variance), the
# last one is mean value.
feats = np.empty([len(FHR), np.floor(len(FHR[0])/SegLen), 8])

for key in FHR:
    fhr = FHR[key]
    T = np.reciprocal(fhr/60.)
    bline = Bline[key]
    DeblineFhr = fhr-bline

    # use ARX model: y(t) = a1*y(t-1) + c0 + c1*t + c2*t^2 + w0, 4 coefficients in total
    AR = np.zeros([np.floor(len(fhr)/SegLen), 4])
    variability = np.zeros([np.floor(len(fhr)/SegLen), 2])
    mean_rr = np.zeros([np.floor(len(fhr)/SegLen), 1])
    mean_fhr = np.zeros([np.floor(len(fhr)/SegLen), 1])
    for i in np.arange(np.floor(len(fhr)/SegLen)):
        seg = DeblineFhr[SegLen*i:SegLen*(i+1)]
        t = T[SegLen*i:SegLen*(i+1)]
        y = seg[order:]
        X = np.ones([SegLen-1, 4])
        X[:, 0] = seg[0:-1]
        X[:, 2] = np.arange(SegLen-order)
        X[:, 3] = np.power(np.arange(SegLen-order), 2)
        AR[i, :] = np.dot(np.linalg.pinv(X), y)
        variability[i, 0] = stv(t)
        variability[i, 1] = np.var(t)
        mean_rr[i] = np.mean(t)
        mean_fhr[i] = np.mean(seg)/10.
    feats[key, :, :] = np.concatenate((AR, variability, mean_rr, mean_fhr), axis=1)
    print '%d finished' % key
np.save('./features/feats', feats)
