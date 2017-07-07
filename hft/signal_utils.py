"""
Utility Functions of Constructing Signals for Research Purpose
"""

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

logger = logging.getLogger(__name__)

# signal construction
# -------------------

with open(os.path.join('config', 'signal.json')) as signal_config_file:
    signal_config = json.load(signal_config_file)
# N_FLEXIBLE_HALF_SECONDS = signal_config['n_flexible_half_seconds']


def order_imbalance_ratio(px, backward_seconds, forward_seconds, index_series):
    """Order Imbalance Ratio
    Reference: Carteaa, Donnellyb and Jaimungal (2015)
    """
    px['order_imbalance_ratio'] = (px['b1_size']-px['s1_size']) / (px['b1_size']+px['s1_size'])
    px = utils.moving_operate(px, 'order_imbalance_ratio', np.mean, backward_seconds, forward_seconds, index_series)
    return px


def get_order_imbalance_column_name(conservative):
    return ('conservative_' if conservative else '') + 'order_flow_imbalance'


def single_order_imbalance(px, conservative=False):
    """Order Flow Imbalance
     Reference: [1] Rama Cont, Kukanov and Stoikov (2011)
                [2] D Shen (2015)

    :param px: pandas data frame
    :param conservative: logical, if True use definition in [1], otherwise use definition in [2]
    :return: pandas data frame with OFI column appended
    """
    px['delta_b1'] = (px.b1 >= px.b1.shift(1)) * px.b1_size - (px.b1 <= px.b1.shift(1)) * px.b1_size.shift(1)
    px['delta_s1'] = (px.s1 <= px.s1.shift(1)) * px.s1_size - (px.s1 >= px.s1.shift(1)) * px.s1_size.shift(1)
    if conservative:
        px.loc[px.b1 < px.b1.shift(1), 'delta_b1'] = 0.0
        px.loc[px.s1 > px.s1.shift(1), 'delta_s1'] = 0.0
    col_name = get_order_imbalance_column_name(conservative)
    px[col_name] = px['delta_b1'] - px['delta_s1']
    px.drop(['delta_b1', 'delta_s1'], axis=1, inplace=True)
    return px


def order_flow_imbalance(px, backward_seconds, forward_seconds, index_series, conservative=False):
    col_name = get_order_imbalance_column_name(conservative)
    if col_name not in px.columns:
        px = single_order_imbalance(px, conservative)
    px = utils.moving_operate(px, col_name, sum, backward_seconds, forward_seconds, index_series)
    return px


def period_return(price_series):
    price_array = np.array(price_series)
    return np.nan if len(price_array) == 1 else (price_array[-1] - price_array[0]) / price_array[0]


def period_tick_move(price_series, tick_size):
    price_array = np.array(price_series)
    return np.nan if len(price_array) == 1 else (price_array[-1] - price_array[0]) / tick_size


def period_mid_move(px, backward_seconds, forward_seconds, tick_size, index_series):
    """
    Compute period price move, price tick move and return
    """
    px = utils.moving_operate(px, 'mid', lambda x: period_tick_move(x, tick_size),
                              backward_seconds, forward_seconds, index_series, 'tick_move')
    px = utils.moving_operate(px, 'mid', period_return, backward_seconds, forward_seconds, index_series, 'return')
    return px


def signal_on_multiple_dates(pxall, func):
    """Compute signal over multiple days

    :param pxall: pandas data frame, price data
    :param func: function to compute one signal
    :return: pandas data frame with signal column appended
    """
    dates = sorted(list(set(pxall.date)))
    logger.info('Computing signal from %s to %s', dates[0], dates[-1])
    px_list = [func(pxall[pxall.date == x].copy()) for x in dates]
    px_enrich = pd.concat(px_list)
    return px_enrich


# signal research / backtesting
# -----------------------------


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


def xy_corr(px, second_list, x_raw_column, y_raw_column='tick_move', winsorize_option=None):
    px_new = px.copy()
    x_column = [utils.get_moving_column_name(x_raw_column, x, 0) for x in second_list]
    y_column = [utils.get_moving_column_name(y_raw_column, 0, x) for x in second_list]
    if winsorize_option is not None:
        for col in x_column:
            px_new[col] = utils.winsorize(px_new[col], winsorize_option['x_prob'], winsorize_option['x_bound'])
        for col in y_column:
            px_new[col] = utils.winsorize(px_new[col], winsorize_option['y_prob'], winsorize_option['y_bound'])
    big_corr = px_new[x_column + y_column].corr()
    corr_mat = big_corr.loc[y_column, x_column]
    return corr_mat


def xx_corr(px, second_list, column_name, row_name):
    column_names = [utils.get_moving_column_name(column_name, x, 0) for x in second_list]
    row_names = [utils.get_moving_column_name(row_name, x, 0) for x in second_list]
    big_corr = px[column_names + row_names].corr()
    corr_mat = big_corr.loc[row_names, column_names]
    return corr_mat


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
