# imports

# basics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import itertools
from datetime import datetime
from time import time

# ml related
from scipy import stats
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class TransitionAnalyse:
    """
    The class holding all information and functions of the dataset. Initialise the class using a pandas data frame and
    then use the class functions to retrieve the information from the data set.
    """

    def __init__(self, df):
        """
        Initialises the class.

        :param df: a pandas.DataFrame variable, imported using the import_csv_filt function in functions.py.
        """
        if isinstance(df, pd.DataFrame):
            if set(['time', 'width', 'rpm', 'rpm_sp', 'torque', 'hz1', 'hz1_sp', 'hz2', 'hz2_sp',
                    'hz3', 'hz3_sp', 'hz4', 'hz4_sp']).issubset(df.columns):

                self.__data = df
                self.__width = np.array(df['width'])
                self.__rpm_sp = np.array(df['rpm_sp'])
            else:
                raise TypeError(
                    'pandas df does not contain the right columns: make sure to import the correct csv file '
                    'containing the right headers using the using the import_csv_filt function.'
                    "Headers must be: ['time', 'width', 'rpm', 'rpm_sp', 'torque', 'hz1', 'hz1_sp', 'hz2', "
                    "'hz2_sp', 'hz3', 'hz3_sp', 'hz4', 'hz4_sp']")

        else:
            raise TypeError('wrong df datatype: must be of type pandas.core.frame.DataFrame, '
                            'created using the import_csv_filt function.')

        self.__width = np.array(df['width'])
        self.__rpm_sp = np.array(df['rpm_sp'])
        self.__resolution = 20  # resolution of the datapoints, do not change unless it is valid to do so.
        self.__seconds_before = 1
        self.__seconds_after = 3

        # variables to be set later, keeping track of variables here
        self.__bin_edges = None
        self.__change_idx_numbers = None
        self.__final_sort_idx = None
        self.__points_b = None
        self.__points_a = None
        self.__jump_dict = None
        self.__jump_time_dict = None
        self.__lin_coef = None
        self.__poly2_coef = None
        self.__jump_times = None

    def get_data(self):
        """
        Returns the current data set.
        :return: The current pandas.dataframe held by the class instance
        """
        return self.__data

    def sort(self):
        """
        Sorts the data set's transitions by from-rpm and to-rpm. This is necessary to do for the retrieval the right
        information from the many transitions.
        :return: int (1), it means success.
        """
        # locations (True/False) at which rpm is about to change (the next idx it will)
        change_idx = self.__rpm_sp[:-1] != self.__rpm_sp[1:]
        # the numbers at which ^
        self.__change_idx_numbers = np.where(change_idx == True)[
            0]  # np.array of the numbers where the rpm is about to change.

        # the from and to values, unsorted
        change_from = self.__rpm_sp[:-1][change_idx]
        change_to = self.__rpm_sp[1:][change_idx]
        # concatenated into one matrix
        from_to_columns = np.concatenate((change_from.reshape(-1, 1), change_to.reshape(-1, 1)), axis=1)

        # using that matrix to sort on 1) from, 2) to - RPM
        self.__final_sort_idx = np.lexsort((from_to_columns[:, 1], from_to_columns[:, 0]))
        # sorted matrix
        from_to_columns_sorted = from_to_columns[self.__final_sort_idx]

        # where jump group changes
        self.__bin_edges = np.where(from_to_columns_sorted[1:, 1] != from_to_columns_sorted[:-1, 1])[0] + 1
        # add outer edges to series
        self.__bin_edges = np.append(np.insert(self.__bin_edges, 0, 0), len(change_from))
        # Where all the bin edges are in the dataset (!)
        bin_edges_ds = self.__change_idx_numbers[self.__final_sort_idx[self.__bin_edges[:-1]]]

        # The idx numbers in which the dataset has the last rpm before change
        final_sort_ds_from = self.__change_idx_numbers[self.__final_sort_idx]
        # The idx numbers in whicht the dataset has the first rpm after change 
        final_sort_ds_to = self.__change_idx_numbers[self.__final_sort_idx] + 1

        return 1

    def fill_dicts(self):
        """
        Fills the jump dictionary. Must run class function "TransitionAnalyse.sort()" first. The jump dictionary has
        the form key = (from_rpm, to_rpm) and item=[mean, mean_normalised, np.array of all jumps]
        :return: int (1), it means success.
        """
        self.__points_b = self.__seconds_before * self.__resolution
        self.__points_a = self.__seconds_after * self.__resolution
        self.__jump_dict = {}

        for i, edge in enumerate(self.__bin_edges):
            if i == 0:
                edge_prev = edge
                continue

            n_jumps = edge - edge_prev
            y = np.zeros((n_jumps, self.__points_a + self.__points_b + 1))

            for j, k in enumerate(range(edge_prev, edge)):
                # getting the right indexes for whole jump
                ds_idx = self.__change_idx_numbers[self.__final_sort_idx[k]]
                start = ds_idx - self.__points_b
                stop = ds_idx + self.__points_a
                # saving curves for mean calculation
                y[j, :] = self.__width[start:stop + 1]
                # plt.plot(x, y[j,:], label=f'{j}', alpha = 0.2)

            # saving edge for next iteration
            edge_prev = edge

            # calculating mean
            mean = np.sum(y, axis=0) / n_jumps
            # normalising for middle 60% of the data, filters some min and max here.
            minioem, maxioem = np.quantile(mean, [0.20, 0.80])
            mean_norm = (mean - minioem) / (maxioem - minioem)

            # filling dicts
            self.__jump_dict[(self.__rpm_sp[ds_idx], self.__rpm_sp[ds_idx + 1])] = (mean, mean_norm, y)

        return 1


    def plot_norm_jumps(self, random=True, min_jump_threshold=2, alpha=0.1):
        """
        Plots all the normalised transitions in a single graph. Normalises for the inner .6 quantile.
        :param random: If True, plots all graphs in random order. Is nice for graph aesthetics.
        :param min_jump_threshold: Int > 0. If set higher then 0, smaller transitions are omitted.
        :param alpha: sets line transparenty in the plots
        :return: int (1), it means success.
        """
        # x-axis time span
        x = np.linspace(-self.__seconds_before, self.__seconds_after, self.__points_a + self.__points_b + 1)

        items = list(self.__jump_dict.items())
        # randomisation is nice for aesthetics of plot
        if random:
            np.random.seed(42)
            np.random.shuffle(items)

        min_count = 0

        for key, line in items:
            b, a = key
            line = line[1]

            if abs(a - b) <= min_jump_threshold:
                min_count += 1
                continue

            plt.plot(x, line, label=f'rpms {b}' + r' $\rightarrow$ ' + f'{a}', alpha=alpha,
                     c=cm.bwr((abs(a - b) - 8) * 22), linewidth=2)

        print(f'{min_count}  jumps smaller then or equal to {min_jump_threshold} REMOVED ')

        plt.xlabel('time after screw rotation change (s)')
        plt.ylabel('bead width (mm)')
        plt.show()

        return 1


    def plot_specific_jumps(self, pairs, alpha=0.3, mean_c=None):
        '''
        plots the individual transitions selected using the pairs parameter. Not normalised.

        :param pairs: list of list pairs of intergers. Represent the from-to rpm pairs you want to plot.
        :param alpha: set line transparity of the individual transitions in the plot
        :param mean_c: sets the color of the mean line in the plot
        :return: int (1), it means succes
        '''
        # x-axis time span
        x = np.linspace(-self.__seconds_before, self.__seconds_after, self.__points_a +
                        self.__points_b + 1)
        items = list(self.__jump_dict.items())
        np.random.seed(42)

        for pair in pairs:
            from_rpm, to_rpm = pair

            lines = self.__jump_dict[(from_rpm, to_rpm)]
            rndm_color = np.random.rand()
            # plot every line
            for line in lines[2]:
                plt.plot(x, line, alpha=alpha, c=cm.prism(rndm_color), linewidth=2)

            # plot mean
            if mean_c is None:
                plt.plot(x, lines[0],
                         label=f'rpms {from_rpm}' + r' $\rightarrow$' + f'{to_rpm}', alpha=1,
                         c=cm.prism(rndm_color), linewidth=3)
            else:
                plt.plot(x, lines[0],
                         label=f'rpms {from_rpm}' + r' $\rightarrow$' + f'{to_rpm}', alpha=1,
                         c=mean_c, linewidth=3)

        plt.legend()
        plt.xlabel('time after screw rotation change (s)')
        plt.ylabel('bead width (mm)')
        plt.show()

    def show_rpm_stats(self):
        '''
        shows the from and to and from-to distributions of the used RPMs in the data set.

        :return: int (1), it means succes
        '''
        # getting the right rpm data from the dataset
        change_idx = self.__rpm_sp[:-1] != self.__rpm_sp[1:]
        change_from = self.__rpm_sp[:-1][change_idx]
        change_to = self.__rpm_sp[1:][change_idx]
        change_step = change_to - change_from

        # histograms of the from/to/step rpms
        sns.histplot(change_from, binwidth=1, kde=True)
        plt.title('Histogram of changes from')
        plt.show()
        # ---
        sns.histplot(change_to, binwidth=1, kde=True)
        plt.title('Histogram of change to')
        plt.show()
        # ---
        sns.histplot(change_step, binwidth=1, kde=True)
        plt.title('Histogram of change steps')
        plt.show()

        # making the from/to heatmap
        min_rpm = min(change_to)
        max_rpm = max(change_to)

        matrix_size = int(max_rpm - min_rpm + 1)
        from_to_matrix = np.zeros((matrix_size, matrix_size))
        missing_matrix = np.zeros((matrix_size, matrix_size))

        # using nice ticks in the heatmap
        ticks = np.linspace(min_rpm, max_rpm, matrix_size)
        ticks = ticks.astype('int')
        size_ticks = np.linspace(0, matrix_size - 1, matrix_size) + .5

        for i, j in zip(change_from, change_to):
            from_to_matrix[int(i - min_rpm), int(j - min_rpm)] += 1
            missing_matrix[int(i - min_rpm), int(j - min_rpm)] = 1

        # plotting the heatmap, annotated the number of occurences inside
        ax = sns.heatmap(from_to_matrix, linewidth=0.5, annot=True)
        plt.title('from-to counter matrix')
        plt.ylabel('RPM from')
        plt.xlabel('RPM to')
        plt.text(2.2, 10.1, 'down jumps', fontsize=20, color='cyan', weight='bold')
        plt.text(9.2, 4.1, 'up jumps', fontsize=20, color='cyan', weight='bold')
        plt.xticks(size_ticks, ticks)
        plt.yticks(size_ticks, ticks)
        plt.show()

        # clearly showing the missing jumps. 
        ax = sns.heatmap(missing_matrix, linewidth=0.5)
        plt.title('missing steps matrix matrix')
        plt.ylabel('RPM from')
        plt.xlabel('RPM to')
        plt.text(2.2, 9.8, 'down jumps', fontsize=20)
        plt.text(9.2, 3.8, 'up jumps', fontsize=20)
        plt.xticks(size_ticks, ticks)
        plt.yticks(size_ticks, ticks)
        plt.show()

        return 1

    def fit_relations(self, lin=True, poly=True, plot=False, add_columns=True):
        '''
        Fits a linear, second degree polynomial or both relations to the width-rpm relation.
        :param lin: if True, fits linear model
        :param poly: if True, fits second degree polynomial model
        :param plot: if True, plots the models to the KDE density plot
        :param add_columns: If True, adds the new expected width columns to the dataframe.
        :return: int (1), it means succes.
        '''

        # kde threshold for filtering points on high kde areas
        kde_thres = 0.05

        reduced_data = np.array([self.__data['rpm'], self.__data['width']])[:, ::10]
        kde = stats.gaussian_kde(reduced_data)
        density = kde(reduced_data)

        if plot:
            # plotting before cleanup
            sc = plt.scatter(reduced_data[0], reduced_data[1], c=density, alpha=0.5)
            plt.colorbar(sc)
            plt.title('all datapoint, colored by kde')
            plt.xlabel('screw rotation (RPM)')
            plt.ylabel('width (mm)')
            plt.show()

        # concatenating and filtering
        kde_data = np.concatenate((reduced_data, density.reshape((1, -1))),
                                  axis=0)[:, np.where(density > kde_thres)]
        # x-axis plot data
        x_pred = np.linspace(np.min(kde_data[0]), np.max(kde_data[0]), 100).reshape((-1, 1))

        # linear model  
        if lin:
            # create linear model
            clf_lin = LinearRegression()
            # train model
            clf_lin.fit(kde_data[0].T, kde_data[1].T)

            # predict linspace
            y_pred = clf_lin.predict(x_pred)

            # saving coeficients:
            # ax + b
            self.__lin_coef = (clf_lin.coef_[0][0], clf_lin.intercept_[0])

            if add_columns:
                self.__data['exp_width_lin'] = self.__data['rpm'] * self.__lin_coef[0] + self.__lin_coef[1]
                self.__data['exp_width_sp_lin'] = self.__data['rpm_sp'] * self.__lin_coef[0] + self.__lin_coef[1]

        # 2nd polynomial fit
        if poly:
            # create 2nd degree model
            poly_model = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression())
            # fit
            poly_model.fit(kde_data[0].T, kde_data[1].T)
            # predict
            y_pred2 = poly_model.predict(x_pred)

            # adding coefficients:
            # ax^2 + bx + c
            self.__poly2_coef = (poly_model['linearregression'].coef_[0][1], poly_model['linearregression'].coef_[0][0],
                                 poly_model['linearregression'].intercept_[0])

            if add_columns:
                self.__data['exp_width_poly'] = self.__data['rpm'] ** 2 * self.__poly2_coef[0] + \
                                                self.__data['rpm'] * self.__poly2_coef[1] + \
                                                self.__poly2_coef[2]
                self.__data['exp_width_sp_poly'] = self.__data['rpm_sp'] ** 2 * self.__poly2_coef[0] + \
                                                   self.__data['rpm_sp'] * self.__poly2_coef[1] + \
                                                   self.__poly2_coef[2]

        if plot:
            sc = plt.scatter(kde_data[0], kde_data[1], c=kde_data[2], alpha=0.5)
            plt.title('high kde areas only')
            if lin:
                plt.plot(x_pred, y_pred, c='red', label='linear fit')
                plt.title('linear fit on high kde areas only')
            if poly:
                plt.plot(x_pred, y_pred2, c='orange', label='2nd degree polynomial fit')
                plt.title('polynomial fit on high kde areas only')
            if lin and poly:
                plt.title('linear and polynomial fit on high kde areas only')
            plt.colorbar(sc)
            plt.legend()
            plt.xlabel('screw rotation (RPM)')
            plt.ylabel('width (mm)')
            plt.show()

        return 1

    def reset_np_data(self):
        '''
        resets the np data class variables if they somehow have been altered. Which can't unless class is rewritten.
        :return: int (1), it means succes
        '''
        self.__width = np.array(self.__data['width'])
        self.__rpm_sp = np.array(self.__data['rpm_sp'])

        return 1

    def give_mean_jump_time(self, per_stepsize=True, plot=True, spread=False):
        """
        Calculates the mean transition time per jump rpm step size. The time is measured from rpm change to when the
        transition has reached 90% of its change in bead width.

        :param per_stepsize: If True, the transition times are calculated by transition rpm step size
        :param plot: If True, the results are plotted
        :param spread: if True, the individual measurements are also shown.
        :return: int (1), it means succes.
        """

        # define thresholds
        stop_threshold = 0.1  # 10%

        min_rpm, max_rpm = np.min(self.__rpm_sp), np.max(self.__rpm_sp)
        max_jump = int(max_rpm - min_rpm)
        rpm_range = list(range(-max_jump, max_jump + 1))
        rpm_range.remove(0)
        rpm_range = np.array(rpm_range)
        self.__jump_time_dict = {}
        per_jumpsize_time_dict = {}

        for rpm in rpm_range:
            per_jumpsize_time_dict[rpm] = []

        # loop thru jumps:
        for key, lines in self.__jump_dict.items():
            b, a = key
            # access all the individual jumps
            times = []
            for line in lines[2]:
                # 1st second mean
                mean_before = np.nanmean(line[:20])
                # last second mean 
                mean_after = np.nanmean(line[-20:])
                # difference before-after
                dba = abs(mean_after - mean_before)

                thres = stop_threshold * dba
                first = np.where(abs(line - mean_after) < thres)[0]
                if len(first) > 0:
                    stop_i = first[0]
                    stop_t = (stop_i - self.__seconds_before * self.__resolution) / self.__resolution
                    times.append([stop_t, stop_i])
                    per_jumpsize_time_dict[a - b].append(stop_t)
                else:
                    print(f'thres = {thres} not met')
                    stop_i = np.nan
                    stop_t = np.nan
            times = np.array(times)
            if not times == np.array([]):
                self.__jump_time_dict[key] = [times, np.nanmean(times[:, 0])]

            # else:
            #     self.__jump_time_dict[key] = [times, np.mean(times[:, 0])]

        sorted_lst_times = np.zeros((30))

        for rpm, rpm_jumptimes in per_jumpsize_time_dict.items():
            if rpm < 0:
                if np.all(rpm_jumptimes != np.nan) and len(rpm_jumptimes) != 0:
                    sorted_lst_times[rpm + 15] = np.nanmean(rpm_jumptimes)
                    if plot and spread:
                        for single_time in rpm_jumptimes:
                            plt.plot(rpm, single_time, linestyle='', markersize=4,
                                     marker='o', alpha=0.1, c='red')
                else:
                    sorted_lst_times[rpm + 15] = np.nan
            else:
                if np.all(rpm_jumptimes != np.nan) and len(rpm_jumptimes) != 0:
                    sorted_lst_times[rpm + 15 - 1] = np.nanmean(rpm_jumptimes)
                    if plot and spread:
                        for single_time in rpm_jumptimes:
                            plt.plot(rpm, single_time, linestyle='', markersize=4,
                                     marker='o', alpha=0.1, c='red')
                else:
                    sorted_lst_times[rpm + 15 - 1] = np.nan

        mask = np.isfinite(sorted_lst_times)
        if plot:
            plt.plot(rpm_range[mask], sorted_lst_times[mask], linestyle='', markersize=8, marker='o')
            plt.title(f'time from screw rotation change untill '
                      f'{int(np.round(100 - 100 * stop_threshold, 0))}% of total width change')
            plt.xlabel('transtion step size (RPM)')
            plt.ylabel('mean jump time (s)')
            plt.ylim(1, 2.5)
            plt.show()

        self.__jump_times = np.array([rpm_range[mask], sorted_lst_times[mask]])

        return 1
