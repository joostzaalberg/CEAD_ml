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
    
    def __init__(self, df):
        
        if isinstance(df, pd.DataFrame):
            if set(['time', 'width', 'rpm', 'rpm_sp', 'torque', 'hz1', 'hz1_sp', 'hz2', 'hz2_sp', 
                    'hz3', 'hz3_sp', 'hz4', 'hz4_sp']).issubset(df.columns):
                
                self.__data = df
                self.__width = np.array(df['width'])
                self.__rpm_sp = np.array(df['rpm_sp'])
            else:
                raise TypeError('pandas df does not contain the right columns: make sure to import the correct csv file ' 
                      'containing the right headers using the using the import_csv_filt function.' 
                      "Headers must be: ['time', 'width', 'rpm', 'rpm_sp', 'torque', 'hz1', 'hz1_sp', 'hz2', " 
                      "'hz2_sp', 'hz3', 'hz3_sp', 'hz4', 'hz4_sp']" )
                
        else:
            raise TypeError('wrong df datatype: must be of type pandas.core.frame.DataFrame, '
                            'created using the import_csv_filt function.')
            
        self.__width = np.array(df['width'])
        self.__rpm_sp = np.array(df['rpm_sp'])
        self.__resolution = 20  # resolution of the datapoints, do not change unless data retrieving method is changed too. 
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
        
    def get_data(self):
        return self.__data
        
    
    def sort(self):
        # locations (True/False) at which rpm is about to change (the next idx it will) 
        change_idx  = self.__rpm_sp[:-1] != self.__rpm_sp[1:]
        # the numbers at which ^
        self.__change_idx_numbers = np.where(change_idx == True)[0] # np.array of the numbers where the rpm is about to change.
        
        # the from and to values, unsorted
        change_from = self.__rpm_sp[:-1][change_idx]
        change_to = self.__rpm_sp[1:][change_idx]
        # concatenated into one matrix
        from_to_columns = np.concatenate((change_from.reshape(-1,1), change_to.reshape(-1,1)), axis = 1)
        
        # using that matrix to sort on 1) from, 2) to - RPM
        self.__final_sort_idx = np.lexsort((from_to_columns[:,1], from_to_columns[:,0]))
        # sorted matrix
        from_to_columns_sorted = from_to_columns[self.__final_sort_idx]
        
        # where jump group changes
        self.__bin_edges = np.where(from_to_columns_sorted[1:, 1] != from_to_columns_sorted[:-1, 1])[0] + 1  
        # add outer edges to series
        self.__bin_edges = np.append(np.insert(self.__bin_edges, 0, 0), len(change_from))
        # Where all the bin edges are in the dataset (!)
        bin_edges_ds = self.__change_idx_numbers[self.__final_sort_idx[self.__bin_edges[:-1]]]  
        
        # print(self.__rpm_sp[bin_edges_ds])
        # print(from_to_columns[self.__final_sort_idx][:10])
        # print('bin_edges are:\n', self.__bin_edges)
        # print('\n\n\n\n------------------ look underneath pls ----------------\nedge sorted (1:from, 2:to) idx for the original dataset!\n\n')

        final_sort_ds_from = self.__change_idx_numbers[self.__final_sort_idx]      # The idx numbers in which the dataset has the last rpm before change
        final_sort_ds_to   = self.__change_idx_numbers[self.__final_sort_idx] + 1  # The idx numbers in whicht the dataset has the first rpm after change 
        
        return 1
    
    def fill_dicts(self):
        self.__points_b = self.__seconds_before * self.__resolution
        self.__points_a = self.__seconds_after  * self.__resolution
        self.__jump_dict = {}
        
        for i, edge in enumerate(self.__bin_edges):
            if i == 0:
                edge_prev = edge
                continue

            n_jumps = edge - edge_prev
            y = np.zeros((n_jumps, self.__points_a + self.__points_b +1))

            for j, k in enumerate(range(edge_prev, edge)):
                # getting the right indexes for whole jump
                ds_idx = self.__change_idx_numbers[self.__final_sort_idx[k]]
                start = ds_idx - self.__points_b
                stop = ds_idx + self.__points_a
                # saving curves for mean calculation
                y[j,:] = self.__width[start:stop+1]
                # plt.plot(x, y[j,:], label=f'{j}', alpha = 0.2)
            
            # saving edge for next iteration
            edge_prev = edge

            # calculating mean
            mean = np.sum(y, axis=0)/n_jumps
            # normalising for middle 60% of the data
            minioem, maxioem = np.quantile(mean, [0.20, 0.80])
            mean_norm = (mean - minioem) / (maxioem - minioem) 
            
            # filling dicts
            self.__jump_dict[(self.__rpm_sp[ds_idx], self.__rpm_sp[ds_idx+1])] = (mean, mean_norm, y)
            
            
        
        
    def plot_norm_jumps(self, random = True, min_jump_threshold = 2, alpha=0.1):
        # x-axis time span
        x = np.linspace(-self.__seconds_before, self.__seconds_after, self.__points_a + self.__points_b+1)
        
        items = list(self.__jump_dict.items())
        # randomisation is nice for aesthetics of plot
        if random:
            np.random.seed(42)
            np.random.shuffle(items)

        min_count = 0

        for key, line in items:
            b, a = key
            line = line[1]

            if abs(a-b) <= min_jump_threshold:
                min_count += 1
                continue
            
            plt.plot(x, line, label = f'rpms {b}' +r' $\rightarrow$ ' + f'{a}', alpha = alpha,
            c = cm.bwr((abs(a - b)-8)*22), linewidth = 2)

        print(f'{min_count}  jumps smaller then or equal to {min_jump_threshold} REMOVED ')

        plt.xlabel('time after rpm change (s)')
        plt.ylabel('bead width (mm)')
        plt.show()
        
    pairsT=list[list[int]]
        
    def plot_specific_jumps(self, pairs: pairsT, alpha=0.4):
        ''' Plot all jumps and mean of specific jumps.'''
        # x-axis time span
        x = np.linspace(-self.__seconds_before, self.__seconds_after, self.__points_a + self.__points_b+1)
        items = list(self.__jump_dict.items())
        np.random.seed(42)
        
        for pair in pairs:
            from_rpm, to_rpm = pair
        
            lines = self.__jump_dict[(from_rpm, to_rpm)]

            # plot every line
            for line in lines[2]:
                plt.plot(x, line, alpha=alpha, c = cm.prism(np.random.rand()), linewidth = 2)

            # plot mean
            plt.plot(x, lines[0], 
                     label = f'rpms {from_rpm}' +r' $\rightarrow$ ' + f'{to_rpm} MEAN', alpha=1,
                     c = 'black', linewidth = 2)
        
        plt.legend()
        plt.xlabel('time after rpm change (s)')
        plt.ylabel('bead width (mm)')
        plt.show()
        
    def show_rpm_stats(self):
        # getting the right rpm data from the dataset
        change_idx  = self.__rpm_sp[:-1] != self.__rpm_sp[1:]
        change_from = self.__rpm_sp[:-1][change_idx]
        change_to   = self.__rpm_sp[1:][change_idx]
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
        size_ticks = np.linspace(0, matrix_size-1, matrix_size)+.5

        for i, j in zip(change_from, change_to):
            from_to_matrix[int(i - min_rpm), int(j-min_rpm)] += 1
            missing_matrix[int(i - min_rpm), int(j-min_rpm)] = 1

        # plotting the heatmap, annotated the number of occurences inside
        ax = sns.heatmap(from_to_matrix, linewidth=0.5, annot=True)
        plt.title('from-to counter matrix')
        plt.ylabel('rpm from')
        plt.xlabel('rpm to')
        plt.text(2.2, 9.8, 'down jumps' , fontsize=20, color='cyan', weight='bold')
        plt.text(9.2, 3.8, 'up jumps' , fontsize=20, color='cyan', weight='bold')
        plt.xticks(size_ticks, ticks)
        plt.yticks(size_ticks, ticks)
        plt.show()

        # clearly showing the missing jumps. 
        ax = sns.heatmap(missing_matrix, linewidth=0.5)
        plt.title('missing steps matrix matrix')
        plt.ylabel('rpm from')
        plt.xlabel('rpm to')
        plt.text(2.2, 9.8, 'down jumps' , fontsize=20)
        plt.text(9.2, 3.8, 'up jumps' , fontsize=20)
        plt.xticks(size_ticks, ticks)
        plt.yticks(size_ticks, ticks)
        plt.show()
        
    
    def fit_relations(self, lin=True, poly=True, plot=False, add_columns = True):
        
        reduced_data = np.array([self.__data['rpm'], self.__data['width']])[:,::10]
        x, y = reduced_data
        kde = stats.gaussian_kde(reduced_data)
        density = kde(reduced_data)
        
        if plot:
            # plotting before cleanup
            sc = plt.scatter(reduced_data[0], reduced_data[1], c=density, alpha = 0.5)
            plt.colorbar(sc)
            plt.title('all datapoint, colored by kde')
            plt.xlabel('screw rpm')
            plt.ylabel('width (mm)')
            plt.show()
        
        # concatenating and filtering
        kde_data = np.concatenate((reduced_data, density.reshape((1,-1))), 
                                  axis=0)[:, np.where(density>0.05)]        
        # x-axis plot data
        x_pred = np.linspace(np.min(kde_data[0]), np.max(kde_data[0]), 100).reshape((-1,1))
        
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
                self.__data['exp_width_lin']    = self.__data['rpm']    * self.__lin_coef[0] + self.__lin_coef[1]
                self.__data['exp_width_sp_lin'] = self.__data['rpm_sp'] * self.__lin_coef[0] + self.__lin_coef[1]
            
        # 2nd polynomial fit
        if poly:
            # create 2nd degree model
            poly_model=make_pipeline(PolynomialFeatures(2, include_bias=False),LinearRegression())
            # fit
            poly_model.fit(kde_data[0].T, kde_data[1].T)
            # predict
            y_pred2 = poly_model.predict(x_pred)

            # adding coefficients:
            # ax^2 + bx + c
            self.__poly2_coef = (poly_model['linearregression'].coef_[0][1], poly_model['linearregression'].coef_[0][0], 
                                 poly_model['linearregression'].intercept_[0])

            if add_columns:
                self.__data['exp_width_poly']    = self.__data['rpm']**2    * self.__poly2_coef[0] + \
                                                   self.__data['rpm']       * self.__poly2_coef[1] + \
                                                   self.__poly2_coef[2]
                self.__data['exp_width_sp_poly'] = self.__data['rpm_sp']**2 * self.__poly2_coef[0] + \
                                                   self.__data['rpm_sp']    * self.__poly2_coef[1] + \
                                                   self.__poly2_coef[2]

        if plot:
            sc = plt.scatter(kde_data[0], kde_data[1], c=kde_data[2], alpha = 0.5)
            plt.title('high kde areas only')
            if lin: 
                plt.plot(x_pred, y_pred, c='red', label = 'linear fit')
                plt.title('linear fit on high kde areas only')
            if poly:
                plt.plot(x_pred, y_pred2, c='orange', label = '2nd degree polynomial fit')
                plt.title('polynomial fit on high kde areas only')
            if lin and poly:
                plt.title('linear and polynomial fit on high kde areas only')
            plt.colorbar(sc)
            plt.legend()
            plt.xlabel('screw rpm')
            plt.ylabel('width (mm)')
            plt.show()
            
    
    def reset_np_data(self):
        '''Reset de numpy data arrays width and rmp_sp, if they are somehow altered or damaged'''
        self.__width  = np.array(self.__data['width' ])
        self.__rpm_sp = np.array(self.__data['rpm_sp'])
        
        return 1
    
    
    def give_mean_jump_time(self, per_stepsize = True):
        
        # define thresholds
        stop_threshold = 0.1  # 10%
        
        min_rpm, max_rpm = np.min(self.__rpm_sp), np.max(self.__rpm_sp)
        max_jump = int(max_rpm - min_rpm)
        rpm_range = list(range(-max_jump, max_jump + 1))
        rpm_range.remove(0)  
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
                mean_before = np.mean(line[ :20])
                # last second mean 
                mean_after  = np.mean(line[-20:])
                dba = abs(mean_after - mean_before)
                
                thres = stop_threshold * dba
                first = np.where(abs(line - mean_after)<thres)[0]
                if len(first)>0:
                    stop_i = first[0]
                    stop_t = (stop_i-self.__seconds_before * self.__resolution) / self.__resolution
                    times.append([stop_t, stop_i])
                    per_jumpsize_time_dict[a-b].append(stop_t)
                else:
                    print(f'thres = {thres} not met')
                    stop_i = np.nan
                    stop_t = np.nan
            times = np.array(times)
            if not times == np.array([]):
                self.__jump_time_dict[key] = [times, np.mean(times[:, 0])]

            # else:
            #     self.__jump_time_dict[key] = [times, np.mean(times[:, 0])]
        
        sorted_lst_times = [None]*30
        
        print(sorted_lst_times)
            
        print('jumptimes are:')
        for rpm, rpm_jumptimes in per_jumpsize_time_dict.items():
            print(f'jump: {rpm}, time: {np.round(np.mean(rpm_jumptimes),2)}')
            if rpm<0:
                sorted_lst_times[rpm+15] = rpm_jumptimes
            else:
                sorted_lst_times[rpm+15-1] = rpm_jumptimes
        
        list_of_mean_jump_times = []
        print(list_of_mean_jump_times)
        
        plt.plot(rpm_range, list_of_mean_jump_times)
        plt.show()