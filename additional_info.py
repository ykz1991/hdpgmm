import pandas as pd
import numpy as np
import scipy.io as sio

columns = ['pH', 'BDecf', 'pCO2','BE','Apgar1','Apgar5','NICU_days','Seizures','HIE','Intubation','Main_diag','Other_diag','Gest_weeks','Weight_g','Sex','Age','Gravidity','Parity','Diabetes','Hypertension','Preeclampsia','Liq_praecox','Pyrexia','Meconium','Presentation','Induced','Istage','NoProgress','CK_KP','IIstage','Deliv_type','dbID','Rec_type', 'Pos_IIst', 'Sig2Birth']

mat = sio.loadmat('../../data/CTU-UHB/params.mat')
mat = mat['params']

NumCol = len(columns)
NumRow = len(mat)
data = np.zeros((NumRow, NumCol))
for i in xrange(NumRow):
    for j in xrange(NumCol):
        data[i][j] = mat[i][0][0][0][j][0][0]
df = pd.DataFrame(data=data, index=range(NumRow), columns=columns)
print df.head()
df.to_csv(open('../../data/CTU-UHB/params.df', 'wb'))
