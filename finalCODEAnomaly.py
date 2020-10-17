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
df1= df1.loc['2016-02-15 11:54:00':'2016-03-08 11:54:00']
df1['Scaled Redox']= df1['Redox'] /8.94
df1['Scaled pH']= df1['pH'] /0.08755
df1['Scaled Cl_2']= df1['Cl_2'] /0.00493
df1['Scaled Cl']= df1['Cl'] /0.00181
df1['Scaled Leit']= df1['Leit'] /25
df1['Scaled Trueb']= df1['Trueb'] /0.005
df1['Scaled Fm']= df1['Fm'] /20.70
df1['Scaled Fm_2']= df1['Fm_2'] /12.48
df1['Scaled Tp']= df1['Tp']/0.101

#dfnew=df1[['Scaled Redox','Scaled pH','Scaled Cl','Scaled Cl_2','Scaled Fm','Scaled Fm_2','Scaled Tp','Scaled Leit','Scaled Trueb']]
#plot(dfnew)


#df1['Scaled Tp']= df1['Tp'] /0.00181

df1['EVENT']= df1['EVENT'].astype(bool)
df2=df1[['Scaled Redox']]
#df5=df1[['Scaled Tp']]
df6=df1[['Scaled Cl']]
df7=df1[['Scaled Cl_2']]
#df8=df1[['Scaled Leit']]
c=df1['EVENT']

#from adtk.detector import PcaAD
#pca_ad = PcaAD(k=1)
#anomalies= pca_ad.fit_detect(df2)
#p=plot(df2, anomaly_pred=anomalies, ts_linewidth=2, ts_markersize=3, ap_color='red', ap_alpha=0.3, curve_group='all');

from adtk.detector import GeneralizedESDTestAD
esd_ad = GeneralizedESDTestAD(alpha=0.3)
anomalies = esd_ad.fit_detect(df2)
q=plot(df2,title='Generalized Extreme studentized Deviate Test on Redox', anomaly_pred=anomalies, ts_linewidth=2, ts_markersize=3, ap_markersize=5, ap_color='red', ap_marker_on_curve=True);

from adtk.detector import GeneralizedESDTestAD
esd_ad = GeneralizedESDTestAD(alpha=0.3)
anomalies1 = esd_ad.fit_detect(df7)
q=plot(df7,title='Generalized Extreme studentized Deviate Test on Cl_2',anomaly_pred=anomalies1, ts_linewidth=3, ts_markersize=3, ap_markersize=5, ap_color='red', ap_marker_on_curve=True);

h=pd.merge(pd.DataFrame(anomalies), pd.DataFrame(anomalies1),left_index=True,right_index=True,how='outer')
from adtk.aggregator import OrAggregator
k=OrAggregator().aggregate(h)
plot(k,title='Aggregated Anomalies plot',ts_color='red',ts_linewidth='3')
plot(c,k,title='Events plot Vs Aggregate Anomalies plot',ts_color='green',ts_linewidth='3')

from sklearn.metrics import confusion_matrix
labels = ['False', 'True']
ConfusionMatrix=confusion_matrix(c,k)
print(ConfusionMatrix)
from sklearn.metrics import classification_report
target_names = ['False', 'True']
print(classification_report(c,k, target_names=target_names))
plt.show()


plt.clf()
plt.imshow(ConfusionMatrix, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('2016-02-15 11:54 to 2016-03-08 11:54- F1 Score 0.94')
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


              
