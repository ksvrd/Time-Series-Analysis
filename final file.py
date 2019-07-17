# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:59:21 2019

@author: THIS PC
"""
import os


import pandas as pd

ada = pd.read_csv('adatepemah.csv')
aki = pd.read_csv('akincilarmah.csv')
ata = pd.read_csv('ataturkmah.csv')
bari = pd.read_csv('barismah.csv')

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

#visualizing individual time series
import matplotlib.pyplot as plt

#renaming price columns to the neighborhood it represents
ada.columns = ['adatepemah']
aki.columns = ['akincilarmah']
ata.columns = ['ataturkmah']
bari.columns = ['barismah']

#merging dataframes
frames = [ada, aki, ata, bari]
df = pd.concat(frames, axis = 1, join = 'outer')

#subplots
ax1= df.plot(subplots = True, 
            figsize = (10, 5),
            marker = '.')

#ADA dataframe
ax_ada = ada.plot(marker = 'o', figsize=(12, 5))
ax_ada.set_xlabel('Date', fontsize=14)
ax_ada.set_ylabel('Price', fontsize=14)
ax_ada.set_title('adatememah')
plt.show()

#AKI dataframe
ax_aki = aki.plot(marker = 'o', figsize=(12, 5), color = 'orange')
ax_aki.set_xlabel('Date', fontsize=14)
ax_aki.set_ylabel('Price', fontsize=14)
ax_aki.set_title('akincilarmah')
plt.show()

#ATA dataframe
ax_ata = ata.plot(marker = 'o', figsize=(12, 5), color = 'green')
ax_ata.set_xlabel('Date', fontsize=14)
ax_ata.set_ylabel('Price', fontsize=14)
ax_ata.set_title('ataturkmah')
plt.show()

#BARI dataframe
ax_bari = bari.plot(marker = 'o', figsize=(12, 5), color = 'red')
ax_bari.set_xlabel('Date', fontsize=14)
ax_bari.set_ylabel('Price', fontsize=14)
ax_bari.set_title('barismah')
plt.show()

#Visualizing multiple time series
ax = df.plot(marker = 'o', figsize=(12, 5))
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('Multiple Plots')
plt.show()

#Calculations May 2019, May 2018, & % change in price, predictions for June, July, August 2019
columns = ['May 2018', 'May 2019', '12 mo. change (%)', 'June 2019 (Pred.)', 'July 2019 (Pred.)', 'August 2019 (Pred.)']
index = ['adatepemah', 'akincilarmah', 'ataturkmah', 'barismah']
data = []
calculations = pd.DataFrame(data, index = index, columns = columns)

#making predictions
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima_model import ARMA

#addressing missing values
df = df.bfill()

#check stationarity
from statsmodels.tsa.vector_ar.vecm import coint_johansen
coint_johansen(df,-1,1).eig

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
pred_df = pd.DataFrame(pred)
pred_df['date'] = pd.to_datetime(['2019-06-01', '2019-07-01', '2019-08-01'], format = '%Y-%m-%d')
pred_df = pred_df.set_index('date')
pred_df.columns = index

#adding predictions to df
df1 = df.append(pred_df)

#Adding Calculations to each column
calculations['May 2018'] = df.loc['2018-05-01']
calculations['May 2019'] = df.loc['2019-05-01']
calculations['12 mo. change (%)'] = (calculations['May 2019'] - calculations['May 2018'])/calculations['May 2018'] *100
calculations['June 2019 (Pred.)'] = pred[0]
calculations['July 2019 (Pred.)'] = pred[1]
calculations['August 2019 (Pred.)'] = pred[2]
print(calculations)

#Visualizing multiple time series with predictions
# Start and end of the date range to extract
legend_values = []
for i, v in enumerate(calculations['August 2019 (Pred.)']):
    index_value = (calculations.index[i] + ':'+'$'+ str(round(v, 2)))
    legend_values.append(index_value)
    
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df1.loc[:,:],
        marker = '.')
ax.legend((legend_values),
          title = 'August 2019',
          loc=2,
          shadow=bool)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Price', fontsize=14)
ax.set_title('Multiple Plots with Predictions')
plt.show()

#adding table with caluclations and predictions to graph

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df1.loc[:,:],
        marker = '.')
ax.legend((legend_values),
          title = 'August 2019',
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