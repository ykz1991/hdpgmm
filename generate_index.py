import numpy as np

pH = np.load('pH.npy')
threshold = 7.1

a = np.where(pH <= threshold)[0]
b = np.where(pH > 7.2)[0][0:len(a)]
idx = np.concatenate((a, b))
np.save('./index/idx_710', idx)
