# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 21:15:36 2019

@author: THIS PC
"""

import os

os.chdir('c:\\Users\\THIS PC\\Desktop\\Weff Files\\timeseriesorder2\\realest\\completed')

import pandas as pd

buca = pd.read_csv('buca.csv')
circo = pd.read_csv('insaatCiroEndeksi.csv')
guven = pd.read_csv('insaatGuvenEndeksi.csv')
mali = pd.read_csv('insaatMaliyetEndeksi.csv')
kon = pd.read_csv('konutfaizoranlari.csv')
tuf = pd.read_csv('tufeizmir.csv')
usdtry = pd.read_csv('usdtry.csv')

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
frames = [buca, circo, guven, kon, mali, tuf, usdtry]
df = pd.concat(frames, axis = 1, join = 'outer')
df = df.fillna(method='ffill')
df = df.fillna(method='backfill')

#check stationarity
from statsmodels.tsa.vector_ar.vecm import coint_johansen
coint_johansen(df,-1,1).eig

#making predictions
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima_model import ARMA

#creating the train and validation set
train = df[:int(0.8*(len(df)))]
valid = df[int(0.8*(len(df))):]

mod = VAR(endog=train)
result = mod.fit()
prediction = result.forecast(result.y, steps=len(valid))

#fitting model on entire dataset
model = VAR(endog=df)
result2 = model.fit()
pred = result2.forecast(result2.y, steps=3)

#converting pred to dataframe before appending to df
index = ['bucaKonutSatisSayilar', 'insaatCircoEndeksi', 'insaatGuvenEndeksi', 'konutfaizoranlari', 'insaatMaliyetEndeksi', 'tufeizmir', 'usdtry']
pred_df = pd.DataFrame(pred)
pred_df['date'] = pd.to_datetime(['2019-07-01', '2019-08-01', '2019-09-01'], format = '%Y-%m-%d')
pred_df = pred_df.set_index('date')
pred_df.columns = index

#adding predictions to df
df1 = df.append(pred_df)

#creating table of 3 mo. predictions
calculations = pd.DataFrame()
calculations['July 2019 (Pred.)'] = pred[0]
calculations['August 2019 (Pred.)'] = pred[1]
calculations['September 2019 (Pred.)'] = pred[2]
calculations.index = index
print(calculations)

#visualizing individual time series
import matplotlib.pyplot as plt

#buca dataframe
ax_buca = buca.plot(marker = 'o', figsize=(12, 5))
ax_buca.set_xlabel('Date', fontsize=14)
ax_buca.set_ylabel('Price', fontsize=14)
ax_buca.set_title('bucaKonutSatisSayilar')
plt.show()

#circo dataframe
ax_circo = circo.plot(marker = 'o', figsize=(12, 5), color = 'orange')
ax_circo.set_xlabel('Date', fontsize=14)
ax_circo.set_ylabel('Price', fontsize=14)
ax_circo.set_title('insaatCircoEndeksi')
plt.show()

#guven dataframe
ax_guven = guven.plot(marker = 'o', figsize=(12, 5), color = 'green')
ax_guven.set_xlabel('Date', fontsize=14)
ax_guven.set_ylabel('Price', fontsize=14)
ax_guven.set_title('insaatGuvenEndeksi')
plt.show()

#kon dataframe
ax_kon = kon.plot(marker = 'o', figsize=(12, 5), color = 'red')
ax_kon.set_xlabel('Date', fontsize=14)
ax_kon.set_ylabel('Price', fontsize=14)
ax_kon.set_title('konutfaizoranlari')
plt.show()

#mali dataframe
ax_mali = mali.plot(marker = 'o', figsize=(12, 5), color = 'violet')
ax_mali.set_xlabel('Date', fontsize=14)
ax_mali.set_ylabel('Price', fontsize=14)
ax_mali.set_title('insaatMaliyetEndeksi')
plt.show()

#tuf dataframe
ax_tuf = tuf.plot(marker = 'o', figsize=(12, 5), color = 'brown')
ax_tuf.set_xlabel('Date', fontsize=14)
ax_tuf.set_ylabel('Price', fontsize=14)
ax_tuf.set_title('tufeizmir')
plt.show()

#usdtry dataframe
ax_usdtry = usdtry.plot(marker = 'o', figsize=(12, 5), color = 'pink')
ax_usdtry.set_xlabel('Date', fontsize=14)
ax_usdtry.set_ylabel('Price', fontsize=14)
ax_usdtry.set_title('usdtry')
plt.show()

#visualizing multiple time series
#Visualizing multiple time series with predictions
legend_values = []
for i, v in enumerate(calculations['September 2019 (Pred.)']):
    index_value = (calculations.index[i] + ':'+'$'+ str(round(v, 2)))
    legend_values.append(index_value)
    
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df1.loc[:,:],
        marker = '.')
ax.legend((legend_values),
          title = 'September 2019',
          loc=2,
          shadow=bool)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('Multiple Plots with Predictions')
plt.show()

#visualizing multiple time series with predictions table
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df1.loc[:,:],
        marker = '.')
ax.legend((legend_values),
          title = 'September 2019',
          loc=2,
          shadow=bool)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Price', fontsize=14)

ax.table(cellText=calculations.values, 
         colWidths=[0.3]*len(calculations.columns), 
         rowLabels=calculations.index, 
         colLabels=calculations.columns, 
         loc='top')
plt.show()