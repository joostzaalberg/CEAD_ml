# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import packages
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # toggle for linux users (PyCharm terminal: pip install pyqt5 )
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import sklearn as sk
import itertools
from datetime import datetime, time
from sklearn.model_selection import train_test_split
# import scripts and functions
from functions import *

print('imports succesfull')



# setting pd print width and rows
pd.options.display.width = 0
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 50)
# if reset is required:
# pd.reset_option('all')

# datetime things
start_date_str = '2022-09-23 12:21:00.000'
end_date_str = '2022-09-23 12:25:00.000'
# defining start and end times in datetime (not strictly necessary (, yet))
start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S.%f')
end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S.%f')

# start import small dataset #########################################################################################

# Defining import variables, _s stands for small
loc_s = '../data/dataset_small_23-09_14.04.45-14.17.30.csv'
start_s = '2022-09-23 14:04:45.000'
end_s = '2022-09-23 14:17:30.000'

# import, filter and outlier replacement
df_s = import_csv_filt(loc_s, start_s, end_s, plot_outliers=False)
# print(df_s)

# plt.plot(df_s.loc[6600:6900, 'time'], df_s.loc[6600:6900, 'bead_width (mm)'], label='bead width (mm)')
# plt.plot(df_s.loc[6600:6900, 'time'], df_s.loc[6600:6900, 'screw_rpm (RPM)'] / 10, label='screw (rpm/10)')
# plt.plot(df_s.loc[6600:6900, 'time'], df_s.loc[6600:6900, 'screw_torque (Nm)'] / 5, label='screw (Nm/5)')

df_s['bead_width (mm)'].plot()



plt.legend()
plt.show()

# works!
df2 = df_add_column_history(df_s, 'screw_rpm (RPM)', 5, steps=2)

# TODO: build beginning of the ML framework!

# look at data how much of a time delay should be included into a data point
# data import: get to np array
# split into training and testing set (np.random.seed(42))





