import seaborn as sns
import os
import pandas as pd
import numpy as np
np.random.seed(1337)
from IPython.display import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from logger.logger import rootLogger

# Reading files
rootLogger.debug("Reading train, test and RUL files")
train = pd.read_csv('CMaps/train_FD001.txt', sep=" ", header=None)
test = pd.read_csv('CMaps/test_FD001.txt', sep=" ", header=None)
RUL = pd.read_csv('CMaps/RUL_FD001.txt', sep=" ", header=None)

# Preprocessing data
rootLogger.debug("Preprocessing")
train.drop(columns=[26,27],inplace=True)
test.drop(columns=[26,27],inplace=True)

columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
train.columns = columns
test.columns = columns

#delete columns with constant values ​​that do not carry information about the state of the unit
train.drop(columns=['Nf_dmd','PCNfR_dmd','P2','T2','TRA','farB','epr'],inplace=True)
test.drop(columns=['Nf_dmd','PCNfR_dmd','P2','T2','TRA','farB','epr'],inplace=True)

RUL.drop(columns=[1], inplace=True)
RUL.columns = ['RUL']

timeincyclestrain = train.groupby('unit_number', as_index=False)['time_in_cycles'].max()
timeincyclestest = test.groupby('unit_number', as_index=False)['time_in_cycles'].max()

train = pd.merge(train, train.groupby('unit_number',
 as_index=False)['time_in_cycles'].max(),
 how='left', on='unit_number')

train.rename(columns={"time_in_cycles_x": "time_in_cycles",
                      "time_in_cycles_y": "max_time_in_cycles"},
             inplace=True)

train['TTF'] = train['max_time_in_cycles'] - train['time_in_cycles']

# Scaling data
rootLogger.debug("Scaling data")
scaler = MinMaxScaler()
ntrain = train.copy()
ntrain.iloc[:,2:19] = scaler.fit_transform(ntrain.iloc[:,2:19])

ntest = test.copy()
ntest.iloc[:,2:19] = scaler.transform(ntest.iloc[:,2:19])

def fractionTTF(dat,q):
    return(dat.TTF[q]-dat.TTF.min()) / float(dat.TTF.max()-dat.TTF.min())
fTTFz = []
fTTF = []
for i in range(train['unit_number'].min(),train['unit_number'].max()+1):
    dat=train[train.unit_number==i]
    dat = dat.reset_index(drop=True)
    for q in range(len(dat)):
        fTTFz = fractionTTF(dat, q)
        fTTF.append(fTTFz)
ntrain['fTTF'] = fTTF

X_train = ntrain.values[:,1:19]
Y_train = ntrain.values[:, 21]
X_test = ntest.values[:,1:19]

# Training the model
rootLogger.debug("Training the model")
regressor = RandomForestRegressor()
regressor.fit(X_train, Y_train)

rootLogger.debug("Predicting")
score = regressor.predict(X_test)

# Converting back the predicted data
rootLogger.debug("Converting back the predicted data")
test = pd.merge(test, test.groupby('unit_number',
    as_index=False)['time_in_cycles'].max(),
    how='left', on='unit_number')
test.rename(columns={"time_in_cycles_x": "time_in_cycles",
                     "time_in_cycles_y": "max_time_in_cycles"}, inplace=True)
test['score'] = score
test = test.T.drop_duplicates().T

def totcycles(data):
    return(data['time_in_cycles'] / (1-data['score']))
test['maxpredcycles'] = totcycles(test)

def RULfunction(data):
    return(data['maxpredcycles'] - data['max_time_in_cycles'])
test['RUL'] = RULfunction(test)

test = test.astype({'unit_number':int, 'time_in_cycles': int, 'max_time_in_cycles':int})

t = test.columns == 'RUL'
ind = [i for i, x in enumerate(t) if x]
predictedRUL = []
for i in range(test.unit_number.min(), test.unit_number.max()+1):
    npredictedRUL=test[test.unit_number==i].iloc[test[test.unit_number==i].time_in_cycles.max()-1,ind]
    predictedRUL.append(npredictedRUL)

# Plotting true RUL vs predicted RUL
# rootLogger.debug("Plotting true RUL vs predicted RUL")
# plt.figure(figsize = (16, 8))
# plt.plot(RUL, color='red')
# plt.plot(predictedRUL, color='blue')
# plt.xlabel('# Unit', fontsize=16)
# plt.xticks(fontsize=16)
# plt.ylabel('RUL', fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(['True RUL','Predicted RUL'], bbox_to_anchor=(0., 1.02, 1., .102),
#  loc=3, mode='expand', borderaxespad=0)
# plt.show()

# Saving the model
def save_model():
    rootLogger.debug("Saving the model")
    import pickle
    model = regressor
    f = open('fw_model', 'wb')
    pickle.dump(model , f)
    f.close()

def save_scaler():
    rootLogger.debug("Saving the scaler")
    import pickle
    f = open('fw_scaler', 'wb')
    pickle.dump(scaler, f)
    f.close()

save_model()
save_scaler()
