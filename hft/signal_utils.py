"""
Utility Functions of Constructing Signals for Research Purpose
"""

import os
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

with open(os.path.join('config', 'signal.json')) as signal_config_file:
    signal_config = json.load(signal_config_file)
N_FLEXIBLE_HALF_SECONDS = signal_config['n_flexible_half_seconds']


def order_imbalance_ratio(px):
    px['order_imbalance_ratio'] = (px['b1_size']-px['s1_size']) / (px['b1_size']+px['s1_size'])
    return px


def volume_order_imbalance(px):
    px['delta_b1'] = 0.0
    idx = px.b1 == px.b1.shift(1)
    px.loc[idx, 'delta_b1'] = px['b1_size'][idx] - px['b1_size'].shift(1)[idx]
    idx = px.b1 > px.b1.shift(1)
    px.loc[idx, 'delta_b1'] = px['b1_size'][idx]
    px.set_value(px.index[0], 'delta_b1', np.nan)

    px['delta_s1'] = 0.0
    idx = px.s1 == px.s1.shift(1)
    px.loc[idx, 'delta_s1'] = px['s1_size'][idx] - px['s1_size'].shift(1)[idx]
    idx = px.s1 < px.s1.shift(1)
    px.loc[idx, 'delta_s1'] = px['s1_size'][idx]
    px.set_value(px.index[0], 'delta_s1', np.nan)

    px['volume_order_imbalance'] = px['delta_b1'] - px['delta_s1']
    px.drop(['delta_b1', 'delta_s1'], axis=1, inplace=True)
    return px


def period_return(px, seconds):
    """Compute the return for every point within a fixed time period going forward or backward

    :param px: pandas data frame, with column return
    :param seconds: integer, could be positive (forward) or negative (backward)
    :return: pandas data frame, augmented by period return
    """
    shift_second_col_name = 'second'+str(seconds)
    px[shift_second_col_name] = px['second'] + seconds

    # there are many missing shifted mids (approximately 50%) so we allow flexible period
    px['orig_shift_second'] = px[shift_second_col_name]
    for i in range(N_FLEXIBLE_HALF_SECONDS):
        px.loc[~px[shift_second_col_name].isin(px.second), shift_second_col_name] = \
            px.loc[~px[shift_second_col_name].isin(px.second), 'orig_shift_second'] + 0.5*(i+1)
        px.loc[~px[shift_second_col_name].isin(px.second), shift_second_col_name] = \
            px.loc[~px[shift_second_col_name].isin(px.second), 'orig_shift_second'] - 0.5*(i+1)
    px.drop('orig_shift_second', axis=1, inplace=True)

    px_copy = px[['second', 'mid']].copy()
    shift_mid_col_name = 'mid'+str(seconds)
    px_copy.columns = [shift_second_col_name, shift_mid_col_name]
    px = pd.merge(px, px_copy, on=shift_second_col_name, how='left')
    shift_return_col_name = 'return'+str(seconds)
    if seconds > 0:
        px['price_change'] = px[shift_mid_col_name] - px.mid
        px[shift_return_col_name] = px['price_change'] / px.mid
    else:
        px['price_change'] = px.mid - px[shift_mid_col_name]
        px[shift_return_col_name] = px['price_change'] / px[shift_mid_col_name]
    return px


def signal_on_multiple_dates(pxall, fun):
    dates = set(pxall.date)
    logger.info('Computing signal from %s to %s', dates[0], dates[-1])
    px_list = [fun(pxall[pxall.date == x].copy()) for x in dates]
    px_enrich = pd.concat(px_list)
    return px_enrich
