import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adtk.detector import PersistAD
from adtk.visualization import plot
from sklearn.ensemble import IsolationForest
import seaborn as sns
from matplotlib import cm

s_train = pd.read_csv("C:/Users/ABB/PycharmProjects/anomaly2/waterDataTraining.csv", index_col="Time", parse_dates=True, squeeze=True)
from adtk.data import validate_series
s_train = validate_series(s_train)

df1=s_train[['Redox','pH','Cl','Cl_2','Fm','Fm_2','Leit','Trueb','Tp','EVENT']]
df1=df1.fillna(df1.mean())
df1= df1.loc['2016-02-15 11:55:00':'2016-05-10 10:47:00']
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
np.random.seed(1234)
msk = np.random.rand(len(df2)) < 0.8
train = df2[msk]
test = df2[~msk]
trainredox=train[['Scaled Redox']]
testredox=test[['Scaled Redox']]
traincl2=train[['Scaled Cl_2']]
testcl2=test[['Scaled Cl_2']]
traincl=train[['Scaled Cl']]
testcl=test[['Scaled Cl']]
testev=test[['EVENT']]
testev['EVENT']=testev['EVENT'].astype(bool)

#from adtk.detector import GeneralizedESDTestAD
#esd_ad = GeneralizedESDTestAD(alpha=0.3)
#anomalies = esd_ad.fit_detect(traincl)
#anomalies = esd_ad.fit(trainredox)
#q=plot(traincl,title='Generalized Extreme studentized Deviate Test on Redox', anomaly_pred=anomalies, ts_linewidth=2, ts_markersize=3, ap_markersize=5, ap_color='red', ap_marker_on_curve=True);
#m=esd_ad.fit_predict(testredox,anomalies)

from adtk.detector import AutoregressionAD
autoregression_ad = AutoregressionAD(c=4.0)
anomalies = autoregression_ad.fit(trainredox) #Fitting the model
m1=autoregression_ad.fit_predict(testredox,anomalies)#predicting the model
anomaliesde = autoregression_ad.fit_detect(testredox)
plot(testredox, anomaly_pred=anomaliesde, ts_linewidth=1, ap_color='red', ap_marker_on_curve=True);


from adtk.detector import AutoregressionAD
autoregression_ad1 = AutoregressionAD(c=4.0)
anomalies1= autoregression_ad.fit(traincl2) #Fitting the model
m=autoregression_ad1.fit_predict(testcl2,anomalies1)##predicting the model.

anomaliesde = autoregression_ad.fit_detect(testcl2)
plot(testcl2, anomaly_pred=anomaliesde, ts_linewidth=1, ap_color='red', ap_marker_on_curve=True);

#esd_ad1 = GeneralizedESDTestAD(alpha=0.3)
#anomalies = esd_ad.fit_detect(traincl)
#anomalies1 = esd_ad1.fit(traincl2)
#q=plot(traincl,title='Generalized Extreme studentized Deviate Test on Redox', anomaly_pred=anomalies, ts_linewidth=2, ts_markersize=3, ap_markersize=5, ap_color='red', ap_marker_on_curve=True);
#m1=esd_ad1.fit_predict(testcl2,anomalies1)

Merged_df=pd.merge(pd.DataFrame(m), pd.DataFrame(m1),left_index=True,right_index=True,how='outer')
from adtk.aggregator import OrAggregator
k=OrAggregator().aggregate(Merged_df)
plot(k,title='Aggregated Anomalies plot',ts_color='red',ts_linewidth='3')
plot(testev,title='Events plot',ts_color='green',ts_linewidth='3')
k=k.fillna('False').astype(bool) #to fill the first missing value


from sklearn.metrics import confusion_matrix
ConfusionMatrix=confusion_matrix(testev,k)
print(ConfusionMatrix)
from sklearn.metrics import classification_report
target_names = ['False', 'True']
print(classification_report(testev,k,target_names=target_names))
plt.show()

plt.clf()
plt.imshow(ConfusionMatrix, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('2016-02-15 11:54 to 2016-05-10 10:47- F1 Score 0.67')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(ConfusionMatrix[i][j]))
plt.show()





