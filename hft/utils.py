"""
Utility functions
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import mstats

logger = logging.getLogger(__name__)

# array/atom manipulation
# -----------------------


def safe_divide(a, b):
    return np.nan if b == 0 else a/b


def winsorize(array, prob, bound):
    """Winsorize an array based on both prob and bound

    :param array: array-like
    :param prob: list of len=2, tail probabilities, example: [0.01, 0.01]
    :param bound: list of len=2, tail bounds, example: [-100, 100]
    :return: array-like, winsorized array
    """
    winsorized_array = pd.Series(mstats.winsorize(array, tuple(prob)))
    winsorized_array[winsorized_array > bound[1]] = bound[1]
    winsorized_array[winsorized_array < bound[0]] = bound[0]
    return winsorized_array.values


# table aggregation
# -----------------


def aggregate(pxall, group, funs, rename_dict=None):
    daily_agg_px = pxall.groupby(['date', group]).agg(funs)
    daily_agg_px.rename(columns=rename_dict, inplace=True)
    daily_agg_px['n_trades'] = pxall.groupby(['date', group]).size()
    agg_px = daily_agg_px.reset_index().groupby(group).median()
    return agg_px


def left_join(df1, df2, key_column):
    """Left join two pandas data frames. Always replace columns in df1 if also presented in df2

    :param df1: pandas data frame
    :param df2: pandas data frame
    :param key_column: list of strings or string
    :return: pandas data frame
    """
    df1 = df1[list(key_column) + df1.columns.difference(df2.columns).tolist()]
    df = pd.merge(df1, df2, on=key_column, how='left')
    return df


# compute a new column based on a period of data
# ----------------------------------------------


def get_moving_column_name(column, backward_seconds, forward_seconds):
    return column + '_' + str(backward_seconds) + '_' + str(forward_seconds)


def get_raw_column_name(moving_column_name):
    words = moving_column_name.split('_')
    return '_'.join(words[:(len(words)-2)])


def get_index_within_period(second, backward_seconds, forward_seconds, px=None):
    logger.info('Getting index within (%s, %s) seconds', str(backward_seconds), str(forward_seconds))
    forward_second = second + forward_seconds
    backward_second = second - backward_seconds
    index_series = [second.index[(second.between(backward_second[i], forward_second[i])).values] for i in second.index]
    idx_col = get_moving_column_name('index_within_period', backward_seconds, forward_seconds)
    if px is not None:
        px[idx_col] = index_series
    logger.info('Finished getting index within (%s, %s) seconds', str(backward_seconds), str(forward_seconds))
    return pd.Series(index_series, index=second.index, name=idx_col)


def get_index_multiple_dates(pxall, backward_seconds, forward_seconds):
    dates = sorted(list(set(pxall.date)))
    logger.info('Getting index from %s to %s', dates[0], dates[-1])
    index_list = [get_index_within_period(pxall.loc[pxall.date == x, 'second'], backward_seconds, forward_seconds)
                  for x in dates]
    index_series = pd.concat(index_list)
    return index_series


def moving_operate(px, column_name, func, backward_seconds, forward_seconds, index_series, new_column_name=None):
    """Compute the moving operation of a column

    :param px: pandas data frame, need to have column column
    :type px: pandas data frame
    :param forward_seconds: int, number of seconds going forward
    :param backward_seconds: int, number of seconds going backward
    :param column_name: string, column name
    :param func: function, could be average, sum or any user-defined operations
    :param new_column_name: string, new column name
    :param index_series: pandas series, index of prevailing observations
    :return: pandas data frame
    """
    if new_column_name is None:
        new_column_name = column_name
    new_column_name = get_moving_column_name(new_column_name, backward_seconds, forward_seconds)
    logger.info('Computing moving operation')
    index_series = index_series[px.index]
    px[new_column_name] = [func(px.loc[idx, column_name]) for idx in index_series]
    logger.info('Finish computing moving operation')
    return px
