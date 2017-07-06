import os
import json
import logging
import numpy as np
import pandas as pd
import pylab
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats.mstats import winsorize
import scipy.stats as stats
import statsmodels.api as sm

import hft.utils as utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

hft_path = os.path.join(os.environ['HOME'], 'dropbox', 'hft')
data_path = os.path.join(hft_path, 'data')
research_path = os.path.join(hft_path, 'research')

# load enriched data
# ------------------

product = 'zn'  # switch between cu and zn
with open(os.path.join(data_path, 'ticksize.json')) as ticksize_file:
    ticksize_json = json.load(ticksize_file)
tick_size = ticksize_json[product]

px = pd.read_pickle(os.path.join(data_path, product+'_enriched.pkl'))

px20131031 = px[px.date == '2013-10-31']
# px = px[np.isnan(px.tick_move_1_0) | (np.abs(px.tick_move_1_0) <= 5)]
px = px[px.date != '2013-10-31']

# signal and return distribution
# ------------------------------

px[['order_flow_imbalance_1_0', 'order_flow_imbalance_2_0', 'order_flow_imbalance_5_0', 'order_flow_imbalance_10_0',
    'order_flow_imbalance_20_0', 'order_flow_imbalance_30_0', 'order_flow_imbalance_60_0',
    'order_flow_imbalance_120_0', 'order_flow_imbalance_180_0', 'order_flow_imbalance_300_0']].describe()

px[['order_imbalance_ratio_1_0', 'order_imbalance_ratio_2_0', 'order_imbalance_ratio_5_0', 'order_imbalance_ratio_10_0',
    'order_imbalance_ratio_20_0', 'order_imbalance_ratio_30_0', 'order_imbalance_ratio_60_0',
    'order_imbalance_ratio_120_0', 'order_imbalance_ratio_180_0', 'order_imbalance_ratio_300_0']].describe()

px[['tick_move_1_0', 'tick_move_2_0', 'tick_move_5_0', 'tick_move_10_0', 'tick_move_20_0', 'tick_move_30_0',
    'tick_move_60_0', 'tick_move_120_0', 'tick_move_180_0', 'tick_move_300_0']][px.date != '2013-10-31'].describe()

px[['tick_move_5_0', 'tick_move_0_10', 'tick_move_0_20']].describe()

def plot_two_hist(px, column, freq1, freq2):
    column1 = utils.get_moving_column_name(column, freq1, 0)
    column2 = utils.get_moving_column_name(column, freq2, 0)
    plt.subplot(1, 2, 1)
    px[column1].hist(bins=100)
    plt.xlabel(column1)
    plt.subplot(1, 2, 2)
    px[column2].hist(bins=100)
    plt.xlabel(column2)
    return

plot_two_hist(px, 'order_flow_imbalance', 60, 300)
plot_two_hist(px, 'order_imbalance_ratio', 60, 300)
plot_two_hist(px, 'tick_move', 60, 300)

px.groupby(np.abs(px.tick_move_0_10)).size()

print(sum(px.tick_move_0_10 == 0) / sum(~np.isnan(px.tick_move_0_10)))  # % no move
print(sum(np.abs(px.tick_move_0_10) >= 1) / sum(~np.isnan(px.tick_move_0_10)))  # % 1 tick move
print(sum(np.abs(px.tick_move_0_10) >= 2) / sum(~np.isnan(px.tick_move_0_10)))  # % 2 tick move

print(sum(px.tick_move_0_20 == 0) / sum(~np.isnan(px.tick_move_0_20)))  # % no move
print(sum(np.abs(px.tick_move_0_20) >= 1) / sum(~np.isnan(px.tick_move_0_20)))  # % 1 tick move
print(sum(np.abs(px.tick_move_0_20) >= 2) / sum(~np.isnan(px.tick_move_0_20)))  # % 2 tick move

# scatter plot
# ------------

def scatter_plot(px, x_column, x_backward, x_forward, y_column, y_backward, y_forward):
    x_column_name = utils.get_moving_column_name(x_column, x_backward, x_forward)
    y_column_name = utils.get_moving_column_name(y_column, y_backward, y_forward)
    regr_data = px[[x_column_name, y_column_name]].dropna()
    x = regr_data[[x_column_name]].values
    y = regr_data[y_column_name].values
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    print('Coefficients: \n', regr.coef_)
    print('R-square: %f' % regr.score(x, y))
    plt.scatter(x, y, marker='o', s=0.1)
    plt.plot(x, regr.predict(x), color='red', linewidth=1)
    plt.xlabel(x_column_name)
    plt.ylabel(y_column_name)
    plt.show()
    return

def plot_two_scatter(px, x_column, y_column, x_b1, x_f1, y_b1, y_f1, x_b2, x_f2, y_b2, y_f2):
    plt.subplot(1, 2, 1)
    scatter_plot(px, x_column, x_b1, x_f1, y_column, y_b1, y_f1)
    plt.subplot(1, 2, 2)
    scatter_plot(px, x_column, x_b2, x_f2, y_column, y_b2, y_f2)
    return

# forward return by signal
plot_two_scatter(px, 'order_imbalance_ratio', 'tick_move', 1, 0, 0, 1, 5, 0, 0, 5)
plot_two_scatter(px, 'order_flow_imbalance', 'tick_move', 60, 0, 0, 60, 300, 0, 0, 300)
plot_two_scatter(px, 'tick_move',  'tick_move', 5, 0, 0, 5, 60, 0, 0, 60)

# signal by signal
plot_two_scatter(px, 'order_imbalance_ratio', 'tick_move', 1, 0, 1, 0, 5, 0, 5, 0)
plot_two_scatter(px, 'order_flow_imbalance', 'tick_move', 60, 0, 60, 0, 300, 0, 300, 0)
plot_two_scatter(px, 'order_flow_imbalance', 'order_imbalance_ratio', 60, 0, 60, 0, 300, 0, 300, 0)

# correlations
# ------------

def xy_corr(px, second_list, column_name):
    column_names = [utils.get_moving_column_name(column_name, x, 0) for x in second_list]
    return_names = [utils.get_moving_column_name('tick_move', 0, x) for x in second_list]
    big_corr = px[column_names + return_names].corr()
    corr_mat = big_corr.loc[return_names, column_names]
    return corr_mat

def xx_corr(px, second_list, column_name, row_name):
    column_names = [utils.get_moving_column_name(column_name, x, 0) for x in second_list]
    row_names = [utils.get_moving_column_name(row_name, x, 0) for x in second_list]
    big_corr = px[column_names + row_names].corr()
    corr_mat = big_corr.loc[row_names, column_names]
    return corr_mat

second_list = [1, 2, 5, 10, 20, 30, 60, 120, 180, 300]
for sec in second_list:
    px = px[(px[utils.get_moving_column_name('tick_move', 0, sec)] <= 10) | np.isnan(px.tick_move_1_0)]
    px = px[(px[utils.get_moving_column_name('tick_move', sec, 0)] <= 10) | np.isnan(px.tick_move_1_0)]

oir_corr = xy_corr(px, second_list, 'order_imbalance_ratio')
ofi_corr = xy_corr(px, second_list, 'order_flow_imbalance')
autocorr = xy_corr(px, second_list, 'tick_move')
oir_corr.to_csv(os.path.join(research_path, 'oir_corr.csv'))
ofi_corr.to_csv(os.path.join(research_path, 'ofi_corr.csv'))
autocorr.to_csv(os.path.join(research_path, 'autocorr.csv'))

oir_ofi = xx_corr(px, second_list, 'order_imbalance_ratio', 'order_flow_imbalance')
oir_return = xx_corr(px, second_list, 'order_imbalance_ratio', 'tick_move')
ofi_return = xx_corr(px, second_list, 'order_flow_imbalance', 'tick_move')
oir_ofi.to_csv(os.path.join(research_path, 'oir_ofi_corr.csv'))
oir_return.to_csv(os.path.join(research_path, 'oir_return_corr.csv'))
ofi_return.to_csv(os.path.join(research_path, 'ofi_return_corr.csv'))

# multivariate regression
# -----------------------

def reg(px, freq_oir, freq_ofi, freq_xreturn, freq_yreturn, show_plot=True, show_inference=True):
    oir_column_name = utils.get_moving_column_name('order_imbalance_ratio', freq_oir, 0)
    ofi_column_name = utils.get_moving_column_name('order_flow_imbalance', freq_ofi, 0)
    xreturn_column_name = utils.get_moving_column_name('tick_move', freq_xreturn, 0)
    yreturn_column_name = utils.get_moving_column_name('tick_move', 0, freq_yreturn)
    regr_data = px[[oir_column_name, ofi_column_name, xreturn_column_name, yreturn_column_name]].dropna()
    regr_data[ofi_column_name] = winsorize(regr_data[ofi_column_name], (0.005, 0.005))
    # regr_data[xreturn_column_name] = winsorize(regr_data[xreturn_column_name], (0.005, 0.005))
    # regr_data[yreturn_column_name] = winsorize(regr_data[yreturn_column_name], (0.005, 0.005))
    x = regr_data[[oir_column_name, ofi_column_name, xreturn_column_name]].values
    y = regr_data[yreturn_column_name].values
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    yhat = regr.predict(x)
    resids = yhat - y
    if show_plot:
        # regression line
        plt.figure(1)
        plt.scatter(yhat, y, marker='o', s=0.1)
        plt.plot(yhat, yhat, color='red', linewidth=1)
        plt.xlabel('Fitted ' + yreturn_column_name)
        plt.ylabel('Observed ' + yreturn_column_name)
        plt.show()
        # residual histogram
        plt.figure(2)
        plt.hist(resids, bins=40)
        plt.title('Histogram of residuals')
        # residual qq plot
        plt.figure(3)
        stats.probplot(resids, dist="norm", plot=pylab)
        plt.title('QQ plot of residuals')
    if show_inference:
        x2 = sm.add_constant(x)
        est = sm.OLS(y, x2)
        est2 = est.fit()
        print(est2.summary())
    return {'r-square': regr.score(x, y), 'beta': regr.coef_, 'residuals': resids}


freq_oir = 1
freq_ofi = 5
freq_xreturn = 2
freq_yreturn = 10

res = reg(px, freq_oir, freq_ofi, freq_xreturn, freq_yreturn, True)


