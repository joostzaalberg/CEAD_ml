# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import packages
import sys
import subprocess
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
print('imports succesfull')


# setting pd print width and rows and matplotlib
pd.options.display.width = 0
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 50)
# if reset is required:
# pd.reset_option('all')

# toggle for linux users (Terminal: pip install pyqt5 )
if sys.platform == "linux" or sys.platform == "linux2":
    print('Linux is detected, matplotlib backend changed to Qt5Agg')
    package = 'pyqt5'
    try:
        import package
    except ImportError as e:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use("TkAgg") # standard for Windows and Mac

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

# plot
df_s.plot.scatter(x='bead_width (mm)', y='screw_rpm (RPM)', alpha=0.5)
plt.show()

# add some history of some columns
to_expand_columns = ['screw_rpm (RPM)']
df_s = df_add_column_history(df_s, to_expand_columns, n_columns=0, steps=2)

X, y = split_to_np_feat_and_ans(df_s)

# TODO: build beginning of the ML framework!
# scale data and normalise data
#


# defining ML constants
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
clf = SGDRegressor(loss='squared_error', penalty='l2', max_iter=1000)
clf = LinearRegression() # try this one
clf.fit(X, y)


y_pred = clf.predict(X_test)

print(np.round(np.stack((y_test, y_pred), axis = 0), 0))



print('r2 score is : ', r2_score(y_test, y_pred))







