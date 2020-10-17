import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adtk.detector import PersistAD
from adtk.visualization import plot
from sklearn.ensemble import IsolationForest

s_train = pd.read_csv("C:/Program Files/InfluxDB/myinfluxdb/waterDataTraining.csv", index_col="Time", parse_dates=True, squeeze=True)
from adtk.data import validate_series
s_train = validate_series(s_train)

df1=s_train[['Redox','pH','Cl','Cl_2','Leit','EVENT']]
df1=df1.fillna(df1.mean())
df1= df1.loc['2016-02-15 11:55:00':'2016-03-08 11:54:00']
df1['Scaled Redox']= df1['Redox'] /8.94
df1['Scaled pH']= df1['pH'] /0.08755
df1['Scaled Cl_2']= df1['Cl_2'] /0.00493
df1['Scaled Cl']= df1['Cl'] /0.00181
df1['Scaled Leit']= df1['Leit'] /25
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
q=plot(df2, anomaly_pred=anomalies, ts_linewidth=2, ts_markersize=3, ap_markersize=5, ap_color='red', ap_marker_on_curve=True);

from adtk.detector import GeneralizedESDTestAD
esd_ad = GeneralizedESDTestAD(alpha=0.3)
anomalies1 = esd_ad.fit_detect(df7)
q=plot(df7, anomaly_pred=anomalies1, ts_linewidth=2, ts_markersize=3, ap_markersize=5, ap_color='red', ap_marker_on_curve=True);


#from adtk.detector import OutlierDetector
#from sklearn.neighbors import LocalOutlierFactor
#outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.05))
#anomalies1 = outlier_detector.fit_detect(df7)
#q=plot(df7, anomaly_pred=anomalies1, ts_linewidth=2, ts_markersize=3, ap_color='red', ap_alpha=0.3, curve_group='all');

h=pd.merge(pd.DataFrame(anomalies), pd.DataFrame(anomalies1),left_index=True,right_index=True,how='outer')
from adtk.aggregator import OrAggregator
k=OrAggregator().aggregate(h)


#from adtk.detector import SeasonalAD
#seasonal_ad = SeasonalAD(c=3.0, side="both")
#anomalies2 = seasonal_ad.fit_detect(df5)
#plot(df5, anomaly_pred=anomalies2, ts_linewidth=1, ap_color='red', ap_marker_on_curve=True);

#h1=pd.merge(pd.DataFrame(k), pd.DataFrame(anomalies2),left_index=True,right_index=True,how='outer')
#k1=OrAggregator().aggregate(h1)

#from adtk.detector import OutlierDetector
#from sklearn.neighbors import LocalOutlierFactor
#outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.05))
#anomalies4 = outlier_detector.fit_detect(df7)
#plot(df7, anomaly_pred=anomalies4, ts_linewidth=2, ts_markersize=3, ap_color='red', ap_alpha=0.3, curve_group='all');

#h12=pd.merge(pd.DataFrame(anomalies), pd.DataFrame(anomalies4),left_index=True,right_index=True,how='outer')
#k12=OrAggregator().aggregate(h12)


df3= df1[['EVENT']]
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(c,k)
from sklearn.metrics import f1_score
m=f1_score(c,k,average='macro')
from sklearn.metrics import accuracy_score
g=accuracy_score(c,k)

