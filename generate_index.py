import numpy as np
import pickle
import matplotlib.pyplot as plt

# ua = pickle.load(open('../../data/CTU-UHB/ua.pickle', 'rb'))
fhr = pickle.load(open('../../data/CTU-UHB/fhr.pickle', 'rb'))

pH = np.load('pH.npy')
threshold = 7.05
duration = 35*60*4
length = 10*60*4

# percent = np.array([1.*np.sum(ua[i][-length:] == 0) / length * 100 for i in xrange(len(ua))])
fhr_length = np.array([len(fhr[i]) for i in fhr])
'''
a = np.where(pH <= threshold)[0]
b = np.where(pH > 7.2)[0][0:len(a)]
idx = np.concatenate((a, b))
np.save('./index/idx_710', idx)
'''
# a = np.where(np.logical_and(np.logical_and(pH <= threshold, percent < 20), fhr_length > duration))[0]
# b = np.where(np.logical_and(np.logical_and(pH > 7.2, percent < 15), fhr_length > duration))[0]
a = np.where(np.logical_and(pH <= threshold, fhr_length > duration))[0]
b = np.where(np.logical_and(pH > 7.2, fhr_length > duration))[0]
idx = np.concatenate((a, b))
'''
for i in idx:
    print 'index is', i
    plt.plot(np.arange(20*60*4)/240., ua[i][-20*60*4:])
    plt.show()
'''
print len(a), len(b)
np.save('./index/idx_705_unbalanced', idx)
