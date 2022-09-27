import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def import_csv(data_path: str):
    """
    This function loads the data only. Basically just pd.read_csv().
    """
    df = pd.read_csv(data_path)
    # print(df)

    return df


def import_csv_filt(data_path: str, start_date: str, end_date: str, outlier_repl=True,
                    plot_outliers=False) -> pd.DataFrame:
    """
    This function loads the data, filters on the given time window and filters and fills outliers with a rolling
    window median.
    """
    # shifting bead parameters
    n = 17  # 17*0.2 = 3.4s =~ 12/360*100
    # Defining outlier detection parameters
    window_size = 8
    outlier_thres_mm = 3

    # reading in data
    df = pd.read_csv(data_path)

    # Filter on given dates
    df = df[(df['time'] > start_date) & (df['time'] < end_date)]

    # reverse dataset to time increasing while going through the dataset
    df = df[::-1]
    # df['index'] = range(0, len(df))
    # df = df.set_index('index')
    df = df.reset_index(drop=True)

    print(df)


    # shift bead data up to compensate for later reading of the bead
    df.loc[:, 'bead_width (mm)'] = df.loc[:, 'bead_width (mm)'].shift(-n)
    df.drop(df.tail(n).index, inplace=True)

    # get indexes of outliers
    outliers_idx = (df['bead_width (mm)'] -
                    df['bead_width (mm)'].rolling(window_size, center=True).median()).abs() > outlier_thres_mm

    if plot_outliers:
        # plot
        plt.plot(df['bead_width (mm)'], label='bead width (mm)')
        plt.plot(df['bead_width (mm)'].loc[outliers_idx], '.r', label='dropped')
        plt.plot(df['bead_width (mm)'].rolling(window_size, center=True).median().loc[outliers_idx], '.y',
                 label='replaced by')
        plt.legend()
        plt.show()

    if outlier_repl:
        # replace
        df.loc[outliers_idx, 'bead_width (mm)'] = df['bead_width (mm)'].rolling(window_size, center=True).median().loc[
            outliers_idx]

    # if smoothing:
    ## smoothing
    # smooth = df_s['bead_width (mm)'].rolling(window=10, win_type='gaussian', center=True).mean(std=10)

    print(df)

    return df

def df_add_column_history(df : pd.DataFrame, column_name : str, n_columns : int, steps=1) -> pd.DataFrame:
    """"
    The function adds n extra data columns of the selected column name. Please note that there are 5 data points in a
    second, so for every second of extra history, there should be 5 columns. Change interval to positive integer to
    increase the step size.
    """
    # deepcopy, otherwise original df gets edited.
    dfc = df.copy(deep=True)

    # first column
    new_name = f'{column_name}_{np.round(steps * 0.2, 1)}s'
    old_name = column_name
    dfc[new_name] = dfc.loc[:, old_name].shift(-steps)
    # remove head
    dfc.drop(dfc.tail(steps).index, inplace=True)

    old_name = new_name

    # then the rest
    for n in range(2, n_columns+1):
        new_name = f'{column_name}_{np.round(steps*n*0.2, 1)}s'

        # add shifted column
        dfc[new_name] = dfc.loc[:, old_name].shift(-steps)
        # remove head
        dfc.drop(dfc.tail(steps).index, inplace=True)

        old_name = new_name

    return dfc

