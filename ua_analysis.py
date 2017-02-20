import numpy as np
import pickle
import matplotlib.pyplot as plt

length = 2400
ua = pickle.load(open('../../data/CTU-UHB/ua.pickle', 'rb'))
percentZero = np.zeros(len(ua))
for k in xrange(len(ua)):
    y = ua[k][-length:]
    numZero = np.sum(y==0)
    percentZero[k] = 100.*numZero/length
    # print '%.2f%% missing values in %i-th recording' % (percentZero, k)

thr = 50
idxPercentZeroOverThr = np.where(percentZero > thr)[0]
percentRecord = 100.*np.sum(percentZero > thr)/len(percentZero)
print '%.2f%% recordings have more than %d%% missing values.' % (percentRecord, thr)

x = np.arange(length)/4.
for i in idxPercentZeroOverThr:
    plt.plot(x, ua[i][-length:])
    plt.xlabel('time [sec]')
    plt.ylabel('UA')
    plt.show()
