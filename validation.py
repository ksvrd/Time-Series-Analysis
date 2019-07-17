# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:16:50 2019

@author: Steff
"""

import os
import numpy as np
import pandas as pd

os.chdir("C:\\Users\\pro 4\\Desktop\\Validation")

ada = pd.read_csv('adatepemah.csv')
aki = pd.read_csv('akincilarmah.csv')
ata = pd.read_csv('ataturkmah.csv')
bari = pd.read_csv('barismah.csv')
buca = pd.read_csv('buca.csv')
circo = pd.read_csv('insaatCiroEndeksi.csv')
guven = pd.read_csv('insaatGuvenEndeksi.csv')
mali = pd.read_csv('insaatMaliyetEndeksi.csv')
kon = pd.read_csv('konutfaizoranlari.csv')
tuf = pd.read_csv('tufeizmir.csv')
usdtry = pd.read_csv('usdtry.csv')

#converting date column to data time 64 object
ada['date'] = pd.to_datetime(ada['date'], format = '%d/%m/%Y')
aki['date'] = pd.to_datetime(aki['date'], format = '%d-%m-%Y')
ata['date'] = pd.to_datetime(ata['date'], format = '%d/%m/%Y')
bari['date'] = pd.to_datetime(bari['date'], format = '%d-%m-%Y')

#set datatime object as index
ada = ada.set_index('date')
aki = aki.set_index('date')
ata = ata.set_index('date')
bari = bari.set_index('date')

#renaming price columns to the neighborhood it represents
ada.columns = ['adatepemah']
aki.columns = ['akincilarmah']
ata.columns = ['ataturkmah']
bari.columns = ['barismah']

#Cleaning dataframes
buca['date'] = buca['year'].astype(str) + "-" + buca['month'].astype(str)
buca = buca.drop(['month', 'year'], axis = 1)
buca.columns = ['bucaKonutSatisSayilar', 'date']
buca['date'] = pd.to_datetime(buca['date'], format = '%Y-%m')
buca = buca.set_index('date')

circo['date'] = circo['year'].astype(str) + "-" + circo['month'].astype(str)
circo = circo.drop(['month', 'year'], axis = 1)
circo.columns = ['insaatCircoEndeksi', 'date']
circo['date'] = pd.to_datetime(circo['date'], format = '%Y-%m')
circo = circo.set_index('date')

guven['date'] = guven['year'].astype(str) + "-" + guven['month'].astype(str)
guven = guven.drop(['month', 'year'], axis = 1)
guven.columns = ['insaatGuvenEndeksi', 'date']
guven['date'] = pd.to_datetime(guven['date'], format = '%Y-%m')
guven = guven.set_index('date')

mali['date'] = mali['year'].astype(str) + "-" + mali['month'].astype(str)
mali = mali.drop(['month', 'year'], axis = 1)
mali.columns = ['insaatMaliyetEndeksi', 'date']
mali['date'] = pd.to_datetime(mali['date'], format = '%Y-%m')
mali = mali.set_index('date')

kon.columns = ['date', 'konutfaizoranlari']
kon['date'] = pd.to_datetime(kon['date'], format = '%d.%m.%Y')
kon = kon.set_index('date')
kon = kon.resample('MS').mean()

tuf['date'] = tuf['year'].astype(str) + "-" + tuf['month'].astype(str)
tuf = tuf.drop(['month', 'year'], axis = 1)
tuf.columns = ['tufeizmir', 'date']
tuf['date'] = pd.to_datetime(tuf['date'], format = '%Y-%m')
tuf = tuf.set_index('date')

usdtry = usdtry.drop(['Pmin', 'Pmax', 'Diff'], axis = 1)
usdtry.columns = ['date', 'usdtry']
usdtry['date'] = pd.to_datetime(usdtry['date'], format = '%d.%m.%Y')
usdtry = usdtry.set_index('date')
usdtry = usdtry.resample('MS').mean()

#merging dataframes
files = [ada, aki, ata, bari, buca, circo, guven, kon, mali, tuf, usdtry]
df = pd.concat(files, axis = 1, join = 'outer')
df = df.fillna(method='ffill')
df = df.fillna(method='backfill')

#making predictions
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima_model import ARMA

#check stationarity
from statsmodels.tsa.vector_ar.vecm import coint_johansen
coint_johansen(df,-1,1).eig

#creating the train and validation set
train = df[:int(0.8*(len(df)))]
valid = df[int(0.8*(len(df))):]

mod = VAR(endog=train)
result = mod.fit()
prediction = result.forecast(result.y, steps=len(valid))

#converting pred to dataframe before appending to df
pred_df = pd.DataFrame(prediction)
pred_df['date'] = pd.to_datetime(valid.index, format = '%Y-%m-%d')
pred_df = pred_df.set_index('date')
pred_df.columns = valid.columns

#Computations
n_log = []
for file in files:
    logs = np.log(file.iloc[0])
    n_log.append(logs)

nlog_values = []
for i in range(11):
    value = n_log[i][0]
    nlog_values.append(value)
        
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

mse = []
for i in range(10):
    mse_all = mean_squared_error(valid.iloc[:, i].values, pred_df.iloc[:, i].values)
    mse.append(mse_all)

rmse = np.sqrt(mse)

mae = []
for i in range(10):
    mae_all = mean_absolute_error(valid.iloc[:, i].values, pred_df.iloc[:, i].values)
    mae.append(mae_all)

r2s = []
for i in range(10):
    r2s_all = r2_score(valid.iloc[:, i].values, pred_df.iloc[:, i].values)
    r2s.append(r2s_all)