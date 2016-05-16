import numpy as np
import pickle
import matplotlib.pyplot as plt


# compute sample entropy
# parameter r is set to be r*std of x
def sampen(x, m, r):
    N = len(x)
    r = r*np.std(x)
    res_0 = 0.
    res_1 = 0.
    for i in xrange(N-m):
        umi_0 = x[i:i+m]
        umi_1 = x[i:i+m+1]
        count_0 = 0
        count_1 = 0
        for j in xrange(i+1, N-m):
            tmp = umi_0 - x[j:j+m]
            if np.linalg.norm(tmp) <= r:
                count_0 += 1
            tmp = umi_1 - x[j:j+m+1]
            if np.linalg.norm(tmp) <= r:
                count_1 += 1
        res_0 += count_0
        res_1 += count_1
    return -np.log(res_1/res_0)


# compute fuzzy entropy
# parameter r is set to be r*std of x
def fuzzyen(x, m, n, r):
    N = len(x)
    r = r*np.std(x)
    res_0 = 0.
    res_1 = 0.
    for i in xrange(N-m):
        si_0 = x[i:i+m] - np.mean(x[i:i+m])
        si_1 = x[i:i+m+1] - np.mean(x[i:i+m+1])
        tmp_0 = 0.
        tmp_1 = 0.
        for j in xrange(i+1, N-m):
            sj_0 = x[j:j+m] - np.mean(x[j:j+m])
            sj_1 = x[j:j+m+1] - np.mean(x[j:j+m+1])
            dij = np.max(np.abs(si_0 - sj_0))
            Dij = np.exp(-np.power(dij, n)/r)
            tmp_0 += Dij
            dij = np.max(np.abs(si_1 - sj_1))
            Dij = np.exp(-np.power(dij, n)/r)
            tmp_1 += Dij
        # tmp *= 2./(N-m-1)
        res_0 += tmp_0
        res_1 += tmp_1
    return -np.log(res_1/res_0)


# compute Higuchi's dimension
def higuchi(x, k):
    N = len(x)
    l = np.zeros(k)
    for h in xrange(k):
        xh = np.array([x[h+i*k] for i in xrange(np.floor((N-h-1)/k).astype(int)+1)])
        l[h] = np.sum(np.abs(xh[1:]-xh[0:-1])) * (N-1.)/((len(xh)-1)*k) / k
    lk = np.mean(l)
    return lk


FHR = pickle.load(open('fhr.npy', 'rb'))
duration = 10
fhr = FHR[0][-60*4*duration:]
# print 'sample entropy ', sampen(fhr, 2, .5)
# print 'fuzzy entropy ', fuzzyen(fhr, 2, 2, .5)
ks = np.array([2**k for k in xrange(1, 10)])
lks = np.zeros(len(ks))
for i, k in enumerate(ks):
    lks[i] = higuchi(fhr, k)
z = np.polyfit(np.log2(ks), np.log2(lks), 1)
print z
f = np.poly1d(z)
x = np.arange(1, 10)
y = f(x)
plt.loglog(ks, lks, 'o', 2**x, 2**y, color='b')
plt.show()
