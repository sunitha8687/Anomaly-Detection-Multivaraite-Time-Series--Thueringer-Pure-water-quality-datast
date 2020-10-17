import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

#from sklearn import preprocessing
#from sklearn.ensemble import IsolationForest


from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('waterDataTraining.csv',index_col='Time', parse_dates=True)
df1= df.loc['2016-02-15 11:54:00':'2016-02-25 23:59:00']
print(df.head())
#df2=df1.resample('H').mean()
#df1['Tp'].plot(figsize=(12,8))
#result = seasonal_decompose(df1['Tp'],model='add',freq=1440)
#result.plot(figsize=(12,8))
#result.seasonal.plot(figsize=(12,8)) #shows the seasonal component in orange which IS NO.OF days lifecycle so m =1

#Buildingf auto arima model to check the best aic value.In this m plays a very importat role. m is the no.of cycles in your seasonal component plot.
output = auto_arima(df1['Tp'], start_p=1, start_q=1, max_p=3, max_q=3, m=1440, start_P=0, seasonal=True, d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
print(output.summary())
print(len(df1))
#splitting train and test to build SARIMAX
train = df1.iloc[:13686]
test = df1.iloc[13686:]
model=SARIMAX(train['Tp'],order=(0,1,1),seasonal_order=(0,0,0,1))
results=model.fit()
print(results.summary())
#doing predictions using SARIMAX
start = len(train)
end = len(train) + len(test) - 1
predictions = results.predict(start,end,type='levels').rename('SARIMA Predictions')
test['Tp'].plot(legend=True,figsize=(12,8))
predictions.plot(legend=True)
plt.show()
#to evaluate the model
from statsmodels.tools.eval_measures import rmse
error = rmse(test['Tp'],predictions)
print("Error rate:", error)
meantest = test['Tp'].mean()
print("Mean of test data in Tp:", meantest)

#forecasting into the future
#model = SARIMAX(df1['Tp'],order=(0,1,1),seasonal_order=(0,0,0,1))
#results=model.fit()
#fcast = results.predict(len(df1),len(df1)+1440,type='levels').rename('SARIMA Forecast')
#df1['Tp'].plot(legend=True,figsize=(12,8))
#fcast.plot(legend=True)
#plt.show()
