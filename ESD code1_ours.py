import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adtk.detector import PersistAD
from adtk.visualization import plot
from sklearn.ensemble import IsolationForest
import seaborn as sns


s_train = pd.read_csv("waterDataTraining.csv", index_col="Time", parse_dates=True, squeeze=True)
from adtk.data import validate_series
s_train = validate_series(s_train)

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
df2=df1[['Scaled Redox','Scaled Cl_2','Scaled Cl', 'Scaled Leit','EVENT']]
#df5=df1[['Scaled Tp']]
#df6=df1[['Scaled Cl']]
df7=df1[['Scaled Cl_2']]
#df8=df1[['Scaled Leit']]
c=df1['EVENT']

#test-train split
#np.random.seed(1234)
#msk = np.random.rand(len(df2)) < 0.8
train = df2.loc['2016-02-15 11:55:00':'2016-04-19 11:54:00'] #Training set
test = df2.loc['2016-04-19 11:55:00':'2016-05-10 11:54:00'] #Testing Set

trainredox=train[['Scaled Redox']] #Trainig the redox
testredox=test[['Scaled Redox']] #test data of redox

traincl2=train[['Scaled Cl_2']] #Training the cl_2
testcl2=test[['Scaled Cl_2']] #test data of cl_2

testev=test[['EVENT']] #just for checking with the event in the test set, just used for CONFUSION MATRIX


testev['EVENT']=testev['EVENT'].astype(bool)

from adtk.detector import GeneralizedESDTestAD
esd_ad = GeneralizedESDTestAD(alpha=0.3)
anomalies = esd_ad.fit(trainredox)  #training redox with esd algorithm
m=esd_ad.fit_predict(testredox,anomalies) #predicting on the test redox

#just for plottingn the anomalies on actual Redox curve
anomalies_redox = esd_ad.fit_detect(testredox)
plot(testredox, anomaly_pred=anomalies_redox, ts_linewidth=1, ap_color='red', ap_marker_on_curve=True);


esd_ad1 = GeneralizedESDTestAD(alpha=0.3)
anomalies1 = esd_ad1.fit(traincl2) #training cl2 with esd algorithm
m1=esd_ad1.fit_predict(testcl2,anomalies1) #predicting on test cl2

#just for plotting the anomalies on actual Cl_2 curve
anomalies_cl2 = esd_ad1.fit_detect(testcl2)
plot(testcl2, anomaly_pred=anomalies_cl2, ts_linewidth=1, ap_color='red', ap_marker_on_curve=True);

#Using OR aggregator
Merged_df=pd.merge(pd.DataFrame(m), pd.DataFrame(m1),left_index=True,right_index=True,how='outer') #putting the anomalies generated altogether in single data frame
from adtk.aggregator import OrAggregator
k=OrAggregator().aggregate(Merged_df) #adding the result of both the column with OR gate
plot(k,title='Aggregated Anomalies plot',ts_color='red',ts_linewidth='3')
plot(testev,title='Events plot',ts_color='green',ts_linewidth='3')

from sklearn.metrics import confusion_matrix
ConfusionMatrix=confusion_matrix(testev,k)
print(ConfusionMatrix)
from sklearn.metrics import classification_report
target_names = ['False', 'True']
print(classification_report(testev,k,target_names=target_names))
plt.show()




