"""
Utility Functions of Constructing Signals
"""

import numpy as np


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
    pass
