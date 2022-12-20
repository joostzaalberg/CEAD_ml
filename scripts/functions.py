# basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# imports related to scoring
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def import_csv(data_path: str):
    """
    This function loads the data only. Basically just pd.read_csv().

    :param data_path: a string leading to the .csv file
    :return: a pandas.dataframe class.
    """
    df = pd.read_csv(data_path)
    # print(df)

    return df


def import_csv_filt(data_path: str, start_date: int, end_date: str, outlier_repl=True,
                    plot_outliers=False, median_filt=True, reset_index=True, correct_hight = False,
                    plot_correction = False, outlier_window = 8, outlier_thres = 3) -> pd.DataFrame:
    """
    This function loads the data, filters on the given time window, reverses the sequence to have the latest point
    first and resets index. Then, it shifts the bead measurement n steps upward to compensate for the difference in
    time because it is measured later, and then it filters and fills outliers with a rolling window median.

    :param data_path: string to the .csv file
    :param start_date: a time integer of where to start. If None is given, it starts at the beginning of the .csv file.
    :param end_date: a time integer of where to stop. If None is given, it ends on the end of the .csv file.
    :param outlier_repl: If True, it replaces outliers
    :param plot_outliers: If True, it shows you which outliers have been replaced by which data points
    :param median_filt: if True, the data is median filtered
    :param reset_index: if True, the data's index is reset
    :param correct_height: if True, the data is corrected for degrading surface.
    :param plot_correction: if True, the height correction is plotted
    :param outlier_window: int hyperparameter, the window size of the outlier median filter
    :param outlier_thres: float hyperparameter, the threshold for when to call an outlier an outlier.
    :return: the newly acquired pandas.DataFrame
    """
    # shiftin bead parameters
    n = 34  # 53*0.05 = 2.35s. Experimentally determined with stopwatch :/  (!=~ 14.9/360*64) 
    
    # Defining outlier detection parameters (seems to work with trial and error)
    window_size = outlier_window
    outlier_thres_mm = outlier_thres
    # median filter params
    filter_length = 20
    head_tails = int(filter_length/2)

    # reading in data
    df = pd.read_csv(data_path)

    # fill nan with last known value.
    df = df.fillna(method='bfill')

    # Filter on given dates
    if isinstance(start_date, str) and isinstance(end_date, float):
        df = df[(df['time'] > start_date) & (df['time'] < end_date)]
    elif isinstance(start_date, str) and end_date == None:
        df = df[(df['time'] < end_date)]
    elif start_date == None and isinstance(end_date, float):
        df = df[(df['time'] < end_date)]
    else:
        print('data set not filtered on time')

    # reverse dataset to time increasing while going through the dataset
    # df = df[::-1] NOT NEEDED WITH NEW DATA LOGGING METHOD
    df = df.reset_index(drop=True)
    
     # make time scale start at 0
    df['time'] = df['time'] - df.loc[0, 'time'] 

    # shift bead data up to compensate for later reading of the bead
    df.loc[:, 'width'] = df.loc[:, 'width'].shift(-n)
    df.drop(df.tail(n).index, inplace=True)

    # get indexes of outliers
    outliers_idx = (df['width'] -
                    df['width'].rolling(window_size, center=True).median()).abs() > outlier_thres_mm

    if plot_outliers:
        # plot
        plt.plot(df['time'], df['width'], label='width')
        plt.plot(df['time'].loc[outliers_idx], df['width'].loc[outliers_idx], '.r', label='dropped')
        plt.plot(df['time'].loc[outliers_idx], df['width'].rolling(window_size, center=True).median().loc[outliers_idx], '.y',
                 label='replaced by')
        plt.legend()
        plt.title('Replaced Outliers')
        plt.xlabel('time (s)')
        plt.ylabel('width (mm)')
        plt.show()

    if outlier_repl:
        # replace
        print('outliers detected and replaced')
        df.loc[outliers_idx, 'width'] = df['width'].rolling(window_size, center=True).median().loc[
            outliers_idx]
    
    if median_filt:
        print('data median filtered')
        df.loc[:,'width'] = df.loc[:, 'width'].rolling(window=filter_length, center=True).median()
    
    if reset_index:
        print('index reset')
        df.drop(df.head(head_tails).index.union(df.tail(head_tails).index), inplace=True)
        df = df.reset_index(drop=True)
    
    fit_lin = LinearRegression().fit(np.array(df['time']).reshape(-1,1),
                                         np.array(df['width']).reshape(-1,1))

    a = fit_lin.coef_[0]
    b = fit_lin.intercept_
    
    if reset_index:
        print('index reset')
        df.drop(df.head(head_tails).index.union(df.tail(head_tails).index), inplace=True)
        df = df.reset_index(drop=True)
    
    if plot_correction:
        timeseries = np.linspace(0, len(df['width'])/filter_length)
        line = timeseries * a + b

        plt.plot(df.loc[:, 'time'], df.loc[:, 'width'], 'darkcyan', label='median filtered data')
        plt.plot(timeseries, line, 'r', label = 'linear fit')
        plt.legend()
        plt.title('Current full data set with non-steady avg bead width')
        plt.xlabel('time (s)')
        plt.ylabel('width (mm)')
        plt.show()

        
    if correct_hight:
        print('linear change over time in avg. width corrected')
        # fit data to lin regression model
        fit_lin = LinearRegression().fit(np.array(df['time']).reshape(-1,1),
                                         np.array(df['width']).reshape(-1,1))

        a = fit_lin.coef_[0]
        b = fit_lin.intercept_
        
        df['width'] = df['width'] - a * df['time']
        
    if plot_correction:
        fit_lin_corr = LinearRegression().fit(np.array(df['time']).reshape(-1,1),
                                              np.array(df['width']).reshape(-1,1))

        a_corr = fit_lin_corr.coef_[0]
        b_corr = fit_lin_corr.intercept_

        print(f'a, b = {a_corr}, {b_corr}')
        line_corr = timeseries * a_corr + b_corr

        plt.plot(df.loc[:, 'time'], df.loc[:, 'width'], 'darkcyan', label='median filtered data')
        plt.plot(timeseries, line_corr, 'r', label = 'linear fit')
        plt.legend()
        plt.title('Current full data set steady avg bead width')
        plt.xlabel('time (s)')
        plt.ylabel('width (mm)')
        plt.show()  
        
    return df


def df_add_column_history(df: pd.DataFrame, column_name: str or list, n_columns: int, steps=1) -> pd.DataFrame:
    """
    The function adds n extra data columns of the selected column name. Please note that there are 5 data points in a
    second, so for every second of extra history, there should be 5 columns. Change steps to positive integer to
    increase the step size. Resets index.

    :param df: pandas.DataFrame
    :param column_name: Column name (str of list(str)) to add history from
    :param n_columns: number of columns of history to add
    :param steps: step size between histories. 1 means every 1 datapoint is saved.
    :return: the newly acquired data frame.
    """
    # deepcopy, otherwise original df gets edited.
    dfc = df.copy(deep=True)

    if n_columns == 0:
        return dfc
    if type(column_name) == str:
        column_name = [column_name]
    if type(column_name) != list or type(column_name[0]) != str:
        raise "error: column_name should be a string or list of strings"

    for name in column_name:
        # first column
        new_name = f'{name}_{np.round(steps * 0.2, 1)}s'
        old_name = name
        dfc[new_name] = dfc.loc[:, old_name].shift(steps)

        # then the rest
        for n in range(2, n_columns + 1):
            old_name = new_name
            new_name = f'{name}_{np.round(steps * n * 0.2, 1)}s'

            # add shifted column
            dfc[new_name] = dfc.loc[:, old_name].shift(steps)

    # remove head
    dfc.drop(dfc.head(steps * n_columns).index, inplace=True)
    dfc = dfc.reset_index(drop=True)

    return dfc


def split_to_np_feat_and_ans(df: pd.DataFrame) -> (np.array, np.array):
    """
    Splits df to X and y np arrays. y containing the width, X the other features

    :param df: pandas.DataFrame containing "width" and other features
    :return: np.array, np.array
    """
    X = df.loc[:, df.columns != 'width'].to_numpy(dtype=float, copy=True)
    y = df.loc[:, 'width'].to_numpy(dtype=float, copy=True)

    return X, y

def give_prediction_score(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Prints the prediction score of the given y_true and y_pred np.array's.
    :param y_true: a np.array of the true width values
    :param y_pred: a np.array of the predicted width values
    :return: int (1), it means success.
    """
    print(  'mean squared error is : ', mean_squared_error(y_true, y_pred))
    print(  '          r2 score is : ', r2_score(y_true, y_pred))
    print('\n      std of error is : ', np.std(y_pred - y_true))

    # plotting histogram 
    n_bins = 100
    _ = plt.hist(y_pred - y_true, bins=n_bins, density=True)
    plt.title('histogram on prediction error')
    plt.show()
    
    return 1

if __name__ == '__main__':
    pass
