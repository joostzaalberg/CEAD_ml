import pandas as pd
import matplotlib.pyplot as plt

def import_csv(data_path: str):
    df = pd.read_csv(data_path)
    # print(df)

    return df


def import_csv_filt(data_path: str, start_date: str, end_date: str, outlier_repl=True,
                    plot_outliers=False) -> pd.DataFrame:
    """
    This function loads the data, filters on the given time window and filters and fills outliers with a rolling
    window median.
    """

    # Defining outlier detection parameters
    window_size = 8
    outlier_thres_mm = 3

    # reading in data
    df = pd.read_csv(data_path)

    # Filter on given dates
    df = df[(df['time'] > start_date) & (df['time'] < end_date)]

    # get indexes of outliers
    outliers_idx = (df['bead_width (mm)'] - df['bead_width (mm)'].rolling(window_size,
                                                                          center=True).median()).abs() > outlier_thres_mm

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
        df['bead_width (mm)'].loc[outliers_idx] = df['bead_width (mm)'].rolling(window_size, center=True).median().loc[
            outliers_idx]

    # ADD A TIME SHIFT TO COMPENSATE FOR LAG IN WIDTH READING!

    # if smoothing:
    ## smoothing
    # smooth = df_s['bead_width (mm)'].rolling(window=10, win_type='gaussian', center=True).mean(std=10)

    return df
