import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest

file_path = r'~/Dropbox/TH_Koln/Semester_3/Case_Studies/Detecting_Anomalies/waterDataTraining.csv'

csvReader = pd.read_csv(file_path)

csvReader.dropna(inplace=True)

csvReader['Time'] = pd.to_datetime(csvReader['Time'])
csvReader.index = csvReader['Time']
print(csvReader.columns)


#
def isolation_forest_anomaly_detection(df,
                                       column_name,
                                       outliers_fraction):

    #Scale the column that we want to flag for anomalies
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(df[[column_name]])
    scaled_time_series = pd.DataFrame(np_scaled)
    # train isolation forest
    model =  IsolationForest(contamination = outliers_fraction, behaviour='new')
    model.fit(scaled_time_series)
    #Generate column for Isolation Forest-detected anomalies
    isolation_forest_anomaly_column = column_name+'_Isolation_Forest_Anomaly'
    df[isolation_forest_anomaly_column] = model.predict(scaled_time_series)
    df[isolation_forest_anomaly_column] = df[isolation_forest_anomaly_column].map( {1: False, -1: True} )
    return df

df_anomaly=isolation_forest_anomaly_detection(df=csvReader,
                                              column_name='Redox',
                                                         outliers_fraction=.0068)

print(df_anomaly['Redox'].loc[df_anomaly['Redox_Isolation_Forest_Anomaly'] == True])

ax = df_anomaly['Redox'].plot()

ax.scatter(df_anomaly['Time'].loc[df_anomaly['Redox_Isolation_Forest_Anomaly'] == True],df_anomaly['Redox'].loc[df_anomaly['Redox_Isolation_Forest_Anomaly'] == True],color='red')

plt.show()

