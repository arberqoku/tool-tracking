# --- third-party ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tabulate
pd.options.display.max_columns = 15
# --- own ---

from datatools import MeasurementDataReader, Tool, Config, MeasurementSeries, Measurement, DataTypes, Action, to_ts_data
from datatools import ACC, GYR, MAG, MIC, POS, VEL
from fhgutils import contextual_recarray_dtype


mytool = "electric_screwdriver"
#mytool = "pneumatic_screwdriver"
#mytool = "pneumatic_rivet_gun"
firstTimeStamp = np.Inf

mdr = MeasurementDataReader(source="tool-tracking-data")
data_dict = mdr.query(query_type=Measurement).filter_by(Tool == mytool).get()
data_raw = data_dict['01']

pd_dict = {}
# 'acc' 'gyr' 'mag' 'mic'
for sensor in data_raw.keys():
    if sensor == 'classes': break
    obs = data_raw[sensor]
    timestamps = obs.t
    features = obs.X
    targets = obs.y
    firstTimeStamp = np.min((firstTimeStamp, timestamps.min()))
    # print(firstTimeStamp)
    colnames = dict(zip(list(range(len(obs.features) + 2)),["TimeStamp"] + obs.features + ['Class']))
    obs_np = np.hstack((timestamps.reshape(-1,1), features, targets.reshape(-1,1)))
    res = pd.DataFrame(obs_np).rename(colnames, axis='columns')
    pd_dict.update({sensor:res})
for sensor in pd_dict.keys():
    pd_dict[sensor]['TimeStamp'] = pd_dict[sensor]['TimeStamp'] - firstTimeStamp

data_df = pd_dict['acc']
# for sensor in ['gyr','mag']:
# for sensor in ['gyr', 'mag', 'mic']:
for sensor in ['gyr']:
    data_df = data_df.merge(pd_dict[sensor], how="outer", on="TimeStamp", sort=True)
    class_merged = np.where(np.isnan(data_df['Class_x']), data_df['Class_y'], data_df['Class_x'])
    if np.isnan(class_merged).any():
        print("PROBLEM - NA CLASS")
        print(np.asarray(np.isnan(class_merged)).nonzero())
        break
    if np.not_equal(class_merged[data_df['Class_y'].dropna().index],
                    data_df['Class_y'][data_df['Class_y'].dropna().index]).any():
        print("PROBLEM - MULTICLASS")
        print(np.asarray(np.not_equal(class_merged[data_df['Class_y'].dropna().index],
                                      data_df['Class_y'][data_df['Class_y'].dropna().index])).nonzero())
        break
    data_df['Class_y'] = class_merged
    data_df = data_df.drop(['Class_x'], axis=1).rename({'Class_y': 'Class'}, axis='columns')
data_df = data_df.loc[data_df['Class'] != -1]



# data.loc[data['Class'] == 8]
# windows = data[['TimeStamp', 'Class']]
# windows

data = data_df.values
window_size = 60
# obs = data[0:10, :]
data_windowed = data[:(data.shape[0] // window_size) * window_size].reshape(-1, window_size, data.shape[1])
X_windowed = data_windowed[:,:,:-1]
y_windowed = data_windowed[:,:,-1]
np.save("EScrew_1_acc-gyr_60stamps_X",X_windowed)
np.save("EScrew_1_acc-gyr_60stamps_y",y_windowed)
# y_cut.shape
# mdr = MeasurementDataReader(source="tool-tracking-data")
# data = mdr.query().filter_by(
#     MeasurementSeries == ['pythagoras-10-20200716'],  # EScrew
#     DataTypes == [ACC, GYR]
# ).get()
# data_dict = mdr.query().filter_by(
#     MeasurementSeries == ['pythagoras-10-20200716'],  # EScrew
#     DataTypes == [ACC, GYR]
# ).get()
#
# Xt, Xc, y = to_ts_data(data_dict, contextual_recarray_dtype)
# windows = data_df[['TimeStamp', 'Class']].loc[data_df['Class'] != 8]
# duration_arr = None
# for i in [0]:
    # For each measurement, calculate the duration of each window
# windows = data[['TimeStamp', 'Class']]
# X = np.array(windows.values[:,:-1])
# t_diff = np.diff(X[:, 0])  # conversion from ms to s?
# t_diff = np.insert(t_diff, 0, 0)  # first diff is zero
# yt = windows.values[:,-1]
#     # duration is the summed diffs of timestamps of consecutive, equal y's
# current_y = -1
# duration = 0
# stamps = 0
# for i in range(len(yt)):
#     previous_y = current_y
#     current_y = yt[i]
#     if current_y == previous_y:
#         duration += t_diff[i]
#         stamps += 1
#     else:  # add to our duration array
#         if duration_arr is None:
#             duration_arr = np.array([[current_y, stamps, duration]])
#         else:
#             duration_arr = np.append(duration_arr, np.array([[previous_y, stamps, duration]]), axis=0)
#         duration = 0
#         stamps = 0
#         previous_y = current_y
# duration_df = pd.DataFrame(duration_arr, columns=["y", "stamps", "duration"])
#
# # duration_df.loc[duration_df["duration"] == max(duration_df["duration"])]
