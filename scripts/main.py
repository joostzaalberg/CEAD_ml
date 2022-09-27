# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import packages
import numpy as np
import matplotlib
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
matplotlib.use('Qt5Agg')  # toggle for linux users (PyCharm terminal: pip install pyqt5 )
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

# import, filter and outlier replacement on bead_width (mm)
df_s = import_csv_filt(loc_s, start_s, end_s, plot_outliers=False)

# choose which columns to keep
columns_to_keep = ['bead_width (mm)', 'screw_rpm (RPM)']
df_s = df_s[columns_to_keep]

# add some history of some columns
to_expand_columns = ['screw_rpm (RPM)']
df_s = df_add_column_history(df_s, to_expand_columns, n_columns=3, steps=1)

# TODO: build beginning of the ML framework!

# look at data how much of a time delay should be included into a data point -> about 22 * dt!
# Should we make it 25? step 1, 3 or 5?
# data import: get to np array
# split into training and testing set (np.random.seed(42))





