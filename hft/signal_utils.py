"""
Utility Functions of Constructing Signals for Research Purpose
"""

import os
import json
import logging
import numpy as np
import pandas as pd

import hft.utils as utils

logger = logging.getLogger(__name__)

with open(os.path.join('config', 'signal.json')) as signal_config_file:
    signal_config = json.load(signal_config_file)
# N_FLEXIBLE_HALF_SECONDS = signal_config['n_flexible_half_seconds']


def order_imbalance_ratio(px, backward_seconds, forward_seconds):
    """
    Order Imbalance Ratio (Shen, 2015)
    """
    px['order_imbalance_ratio'] = (px['b1_size']-px['s1_size']) / (px['b1_size']+px['s1_size'])
    px = utils.moving_operate(px, 'order_imbalance_ratio', np.mean, backward_seconds, forward_seconds)
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


def order_flow_imbalance(px, backward_seconds, forward_seconds, conservative=False):
    if 'order_flow_imbalance' not in px.columns:
        px = single_order_imbalance(px, conservative)
    col_name = get_order_imbalance_column_name(conservative)
    px = utils.moving_operate(px, col_name, sum, backward_seconds, forward_seconds)
    return px


def period_return(price_series):
    price_array = np.array(price_series)
    return np.nan if len(price_array) == 1 else (price_array[-1] - price_array[0]) / price_array[0]


def period_tick_move(price_series, tick_size):
    price_array = np.array(price_series)
    return np.nan if len(price_array) == 1 else (price_array[-1] - price_array[0]) / tick_size


def period_mid_move(px, backward_seconds, forward_seconds, tick_size):
    """
    Compute period price move, price tick move and return
    """
    px = utils.moving_operate(px, 'mid', lambda x: period_tick_move(x, tick_size),
                              backward_seconds, forward_seconds, 'tick_move')
    px = utils.moving_operate(px, 'mid', period_return, backward_seconds, forward_seconds, 'return')
    return px


def signal_on_multiple_dates(pxall, signal):
    """Compute signal over multiple days

    :param pxall: pandas data frame, price data
    :param signal: function to compute one signal
    :return: pandas data frame with signal column appended
    """
    dates = sorted(list(set(pxall.date)))
    logger.info('Computing signal from %s to %s', dates[0], dates[-1])
    px_list = [signal(pxall[pxall.date == x].copy()) for x in dates]
    px_enrich = pd.concat(px_list)
    return px_enrich
