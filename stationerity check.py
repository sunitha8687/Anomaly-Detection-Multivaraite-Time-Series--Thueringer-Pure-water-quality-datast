import pandas as pd
import matplotlib.pyplot as plt

#from sklearn import preprocessing
#from sklearn.ensemble import IsolationForest
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller

#file_path = r'~/Dropbox/TH_Koln/Semester_3/Case_Studies/Detecting_Anomalies/waterDataTraining.csv'

csvReader = read_csv('waterDataTraining.csv')
csvReader.index = csvReader['Time']
#print(csvReader.describe())
csvReader.dropna(inplace=True)
csvReader['Time'] = pd.to_datetime(csvReader['Time'])
#window = 1440*21 #checking the stationarity of the variable in the data for one week
#window2 = 1440 #checking the stationarity of the variable in the data for daily
X = csvReader['Cl_2']
plt.figure(figsize=(9, 3))
rol_mean = X.rolling('5D').mean()
print(rol_mean)
rol_std = X.rolling('5D').std()
#plt.subplot(121)
X.plot()
rol_mean.plot()
rol_std.plot()

plt.title('Rolling mean and rolling standard deviation plot')
#plt.subplot(122)
#plt.legend(['Original series','Rolling mean','Rolling standard deviation'])
plt.title('Original Series, Rolling mean and rolling standard deviation plot')
plt.show()
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))