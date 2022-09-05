# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# import scripts and functions
from functions import import_csv

data_path = 'data/[test_data]_dashboard_temperatures_2022-08-06T00-00-00_2022-09-04T23-59-59.csv'

if __name__ == '__main__':

    # print(f'sin(5) using numpy = {np.round(np.sin(5), 3)}')
    #
    # print(torch.randn(5,4))
    # import data
    df = import_csv(data_path)

    # do some outlier removal (manually)
    df1 = df[df['HZ1'] < 500]

    # reverse df dataset to get
    df1 = df1[::-1]

    # plotting
    df1.plot.line(x='time', y='HZ1')
    plt.show()