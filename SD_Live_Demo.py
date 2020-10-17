import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from pandas.io.json import json_normalize #package for flattening json in pandas df
import statistics
from influxdb import InfluxDBClient
import time
import datetime

client = InfluxDBClient(host ='localhost', port = 8086)
client.switch_database('waterdatanew4')

init = 0
init_2 = 0
init_3 = 0

while True:
    recent_1 = urllib.request.urlopen("http://localhost:9092/kapacitor/v1/tasks/water_data/recent").read()
    recent_2 = urllib.request.urlopen("http://localhost:9092/kapacitor/v1/tasks/water_data/recent").read()

    while recent_1 == recent_2:
        print("No New Data")
        time.sleep(5)
        recent_2 = urllib.request.urlopen("http://localhost:9092/kapacitor/v1/tasks/water_data/recent").read()

    Detection_Column = 'Cl_2'
    jsonReader = pd.read_json(recent_2)

    jsonReaderc = str(json_normalize(jsonReader['series'], 'columns')).split()
    jsonReaderr = json_normalize(jsonReader['series'], record_path='values')
    jsonReaderr.columns = jsonReaderc[2::2]
    jsonReaderr.index = jsonReaderr['time']


    #print(jsonReaderr.columns)
    # jsonReaderr = jsonReaderr.drop(columns='time')

    #data_recent = jsonReaderr
    data_recent = jsonReaderr.iloc[len(jsonReaderr)-60*12:len(jsonReaderr)]

    values_all = jsonReaderr[Detection_Column]

    if init == 0:
        history = jsonReaderr
        #print(history)
        init = 1
    else:
        history = history.append(jsonReaderr)
        #print((values_all))

    values = data_recent[Detection_Column]
    avg = statistics.mean(values_all)
    std = statistics.stdev(values_all)
    #print(avg, std)
    dev = 3

    #
    outliers = data_recent.loc[abs(values_all -avg)   >  dev*std]

    if init_2 == 0:
        history_outliers = outliers
        print(history_outliers)
        init_2 = 1
        #history_outliers = history_outliers.dropna(inplace=True)
    else:
        history_outliers = history_outliers.append(outliers)
        print((history_outliers))
        print('########## Detected Outliers ######### ')
        print(avg, std)
        print(len(history_outliers))
        #history_outliers.dropna(inplace=True)


    #inf = outliers.to_json()
    #print(len(outliers))


    for i in range(len(outliers)):
        json_body = [
            {
                "measurement": "waterDataOutliers",
                "time": outliers.iloc[i]['time'],
                "fields": {
                    "Tp": outliers.iloc[i]['Tp'],
                    "Cl": outliers.iloc[i]['Cl'],
                    "pH": outliers.iloc[i]['pH'],
                    "Redox": outliers.iloc[i]['Redox'],
                    "Leit": outliers.iloc[i]['Leit'],
                    "Trueb": outliers.iloc[i]['Trueb'],
                    "Cl_2": outliers.iloc[i]['Cl_2'],
                    "Fm": outliers.iloc[i]['Fm'],
                    "Fm_2": outliers.iloc[i]['Fm_2'],
                    "EVENT": outliers.iloc[i]['EVENT']
                }
            }
        ]
        #print(json_body)
        #client.write_points(json_body)

  #  fig = plt.figure('1')
   # ax = fig.add_subplot(1, 1, 1)

    # if init_3 == 0:
    #     fig = plt.figure('1',figsize=(15, 6), dpi=80)
    #     ax = fig.add_subplot(1, 1, 1)
    #     init_3 == 1

    #ax = plt.gca()
    #plt.clf()
    #plt.cla()
    #history['time'] = [pd.to_datetime(d) for d in history['time']]

    #history['time'] = sorted(history['time'])
    #ax.set_xlim([datetime.date(2019, 2, 17), datetime.date(2019, 2, 20)])
    #ax.plot( history['time'], history[Detection_Column] )

    #plt.plot_date(history['time'], history[Detection_Column], c='black')

    #fig.autofmt_xdate()
    #plt.ion()
    #plt.draw()
    #plt.pause(0.001)

    #history_outliers['time'] = [pd.to_datetime(d) for d in history_outliers['time']]
    #history_outliers['time'] = sorted(history_outliers['time'])


    #ax.set_xlim([datetime.date(2019, 2, 17), datetime.date(2019, 2, 20)])
    #ax.plot_date(history_outliers['time'], history_outliers[Detection_Column], c='red')
    #plt.show()


