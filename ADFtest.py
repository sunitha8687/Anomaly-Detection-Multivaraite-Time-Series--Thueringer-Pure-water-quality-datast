import pandas as pd
import matplotlib.pyplot as plt

#from sklearn import preprocessing
#from sklearn.ensemble import IsolationForest
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller

#file_path = r'~/Dropbox/TH_Koln/Semester_3/Case_Studies/Detecting_Anomalies/waterDataTraining.csv'

csvReader = read_csv('waterDataTraining.csv')

print(csvReader.describe())

csvReader.dropna(inplace=True)
csvReader['Time'] = pd.to_datetime(csvReader['Time'])
csvReader.index = csvReader['Time']
window = 1440*7 #checking the stationarity of the variable in the data for one week
window2 = 1440 #checking the stationarity of the variable in the data for daily

X = csvReader['Redox'][:window2]
X1 = csvReader['Cl_2'][:window]
X11 = csvReader['Cl_2'][:window2]
X2 = csvReader['Tp'][:window]
X22 = csvReader['Tp'][:window2]
X3 = csvReader['Cl'][:window]
X33 = csvReader['Cl'][:window2]
X4 = csvReader['pH'][:window]
X5 = csvReader['Leit'][:window]
X55 = csvReader['Leit'][:window2]
X6 = csvReader['Trueb'][:window]
X7 = csvReader['Fm'][:window]
X8 = csvReader['Fm_2'][:window]
print(type(X2))
X.plot()
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

##STL Decomposition for non-stationary data - Tp in our case
import statsmodels.api as sm
res = sm.tsa.seasonal_decompose(X8,freq=1440)
resplot = res.plot()
plt.show()
