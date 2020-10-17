import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adtk.detector import PersistAD
from adtk.visualization import plot
from sklearn.ensemble import IsolationForest
import seaborn as sns
from adtk.detector import GeneralizedESDTestAD
from adtk.aggregator import OrAggregator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from adtk.detector import PcaAD




s_train = pd.read_csv("C:/Program Files/InfluxDB/myinfluxdb/waterDataTraining.csv", index_col="Time", parse_dates=True, squeeze=True)
from adtk.data import validate_series
s_train = validate_series(s_train)
esd_ad = GeneralizedESDTestAD(alpha=0.3)
esd_ad1 = GeneralizedESDTestAD(alpha=0.3)

df1=s_train[['Redox','pH','Cl','Cl_2','Fm','Fm_2','Leit','Trueb','Tp','EVENT']]
df1=df1.fillna(df1.mean())
#df1= df1.loc['2016-02-15 11:55:00':'2016-05-10 11:54:00']
df1['Scaled Redox']= df1['Redox'] /df1['Redox'].max()
df1['Scaled pH']= df1['pH']/df1['pH'].max()
df1['Scaled Cl_2']= df1['Cl_2'] /df1['Cl_2'].max()
df1['Scaled Cl']= df1['Cl'] /df1['Cl'].max()
df1['Scaled Leit']= df1['Leit'] /df1['Leit'].max()
df1['Scaled Trueb']= df1['Trueb'] /df1['Trueb'].max()
df1['Scaled Fm']= df1['Fm'] /df1['Fm'].max()


df1['EVENT']= df1['EVENT'].astype(bool)
df2=df1[['Scaled Redox','Scaled Cl_2','Scaled Cl','Scaled pH','Scaled Leit','EVENT']]

nrows = df2.shape[0]
rng = range(0,nrows-4*1440,1440)
allk = pd.DataFrame()
for index in rng:
     #train = df2[(index):(index+1*1440)]
     #test = df2[(index+1*1440):(index+2*1440)]
     train = df2[(index):(index+4*1440)] #Training on past  4 days
     test = df2[(index+4*1440):(index+5*1440)] #Testing on 1 day
     traindata1=train[['Scaled Redox','Scaled pH']]
     testdata1=test[['Scaled Redox','Scaled pH']]
     traindata2=train[['Scaled Redox','Scaled Cl_2']]
     testdata2=test[['Scaled Redox','Scaled Cl_2']]
     anomalies= esd_ad.fit(traindata2)  #training redox and Cl_2 with esd algorithm
     m=esd_ad.fit_predict(testdata2,anomalies) #predicting on the test redox and Cl_2 with pca algorithm
     pca_ad = PcaAD(k=2)
     anomalies1 = pca_ad.fit(traindata1)  #training redox and PH with pca algorithm
     m1=pca_ad.fit_predict(testdata1,anomalies1) #predicting on the test redox and pH with PCA algorithm 
     f=pd.merge(pd.DataFrame(m), pd.DataFrame(m1),left_index=True,right_index=True,how='outer') #Merging the anomalies in both the algorithm in same data frame
     k=OrAggregator().aggregate(f) #ORing the result of both
     allk = pd.concat([allk,k])

ConfusionMatrix=confusion_matrix(df2['EVENT'].astype(bool)[4*1440:(index+5*1440)],allk.astype(bool))
print(ConfusionMatrix)
target_names = ['False', 'True']
print(classification_report(df2['EVENT'].astype(bool)[4*1440:(index+5*1440)],allk.astype(bool),target_names=target_names))
plot(allk,title='Aggregated Anomalies plot',ts_color='red',ts_linewidth='3')
plot(df2['EVENT'],title='Events plot',ts_color='green',ts_linewidth='3')








