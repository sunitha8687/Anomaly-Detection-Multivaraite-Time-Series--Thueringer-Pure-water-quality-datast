import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

#from sklearn import preprocessing
#from sklearn.ensemble import IsolationForest
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('waterDataTraining.csv',index_col='Time', parse_dates=True)
df1= df.loc['2016-02-15 11:54:00':'2016-02-25 23:59:00']
#df1=df.resample('H').mean()
df1.dropna(inplace=True)
x = df1['Tp']
print(x.head())
#stepwise_fit = auto_arima(x,start_p=0,start_q=0,max_p=6,max_q=3,seasonal=True,trace=True) #trace will show u the arima models that auto_Arima will try to fit
stepwise_fit = auto_arima(x, start_p=1, start_q=1, max_p=3, max_q=3, m=1, start_P=0, seasonal=True, d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
print(stepwise_fit.summary())

#input_series = np.asarray(x)
#model_arima = ARIMA(input_series, order=(0,1,1))
#model_fit = model_arima.fit(disp=0)
#print(model_fit.summary())

