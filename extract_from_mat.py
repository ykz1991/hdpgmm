import scipy.io as sio
import pickle

mat = sio.loadmat('../../data/CTU-UHB/CTUdata.mat')
data = mat['CTUdata']

# matlab cell file has entries: rawFhr, fhr, RawUa, ua, delBeats, TimeUntilEnd

# Extract ua data
data = {}
for i in xrange(len(data)):
    data[i] = data[i][0][0][0][1].T[0]
pickle.dump(data, open('../../data/CTU-UHB/fhr.pickle', 'wb'))