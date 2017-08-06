"""
Backtest Strategy
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn import linear_model

import hft.utils as utils
import hft.signal_utils as signal

logger = logging.getLogger(__name__)


def select_feature(train, config):
    """Select features to fit model

    :param train: pandas data frame
    :param config: dictionary, config parameters
    :return: list of strings, column names
    """
    y_column = utils.get_moving_column_name(config['response_column'], 0, config['holding_period'])
    selected_features = []
    for feature in config['feature_column']:
        logger.debug('Computing correlation of %s and %s', feature, config['response_column'])
        winsorize_option = {'x_prob': config['feature_winsorize_prob'][feature],
                            'x_bound': config['feature_winsorize_bound'][feature],
                            'y_prob': config['response_winsorize_prob'],
                            'y_bound': config['response_winsorize_bound']
                            }
        corr_mat = signal.xy_corr(train, config['feature_freq'], feature, config['response_column'], winsorize_option)
        correlation = corr_mat.loc[y_column]
        selected_features.append(correlation.argmax())
    return selected_features


def fit(train, features, config):
    """Fit linear model using features

    :param train: pandas data frame, must contain columns in features
    :param features: list of column names
    :param config: dictionary, config parameters
    :return: sklearn model class
    """
    y_column = utils.get_moving_column_name(config['response_column'], 0, config['holding_period'])
    regr_data = train[features+[y_column]].dropna()

    # data processing
    for feature in features:
        raw_feature = utils.get_raw_column_name(feature)
        regr_data[feature] = utils.winsorize(regr_data[feature], config['feature_winsorize_prob'][raw_feature],
                                             config['feature_winsorize_bound'][raw_feature])
    regr_data[y_column] = utils.winsorize(regr_data[y_column], config['response_winsorize_prob'],
                                          config['response_winsorize_bound'])
    x = regr_data[features].values
    y = regr_data[y_column].values
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x, y)
    return regr


def backtest(px, config):
    logger.info('Start backtesting')
    dates = list(set(px.date))
    dates.sort()
    y_name = utils.get_moving_column_name(config['response_column'], 0, config['holding_period'])
    btdf = pd.DataFrame()
    columns = ['dt', 'date', 'time', 'price', 'qty', 'volume', 'open_interest',
               'b1', 'b1_size', 's1', 's1_size', 'mid', 'second']
    for i in range(config['training_period'], len(dates)):
        date = dates[i]
        logger.info('Backtesting on %s', date)
        logger.debug('Selecting feature')
        train = px[(px.date >= dates[i-config['training_period']]) & (px.date < date)].copy()
        features = select_feature(train, config)
        logger.debug('Fitting model')
        model = fit(train, features, config)
        logger.debug('Predicting future return')
        px_i = px.loc[px.date == date, columns + features + [y_name]].copy()
        x_new = px_i[features]
        x_new = x_new.fillna(x_new.median())
        alpha = model.predict(X=x_new)
        px_i['alpha'] = alpha
        btdf = btdf.append(px_i)
    logger.info('Finish backtesting')
    return btdf


def trade(btdf, config):
    logger.info('Making trading decision')
    btdf['trade'] = 0
    btdf.loc[btdf.alpha > config['trade_trigger_threshold'][1], 'trade'] = 1
    btdf.loc[btdf.alpha < config['trade_trigger_threshold'][0], 'trade'] = -1
    btdf.loc[btdf.second > config['end_second'], 'trade'] = 0
    btdf.loc[btdf.second < config['start_second'], 'trade'] = 0
    return btdf


def get_fixed_period_close_second(btdf, config):
    btdf['close_second'] = btdf.second + config['holding_period']
    dates = list(set(btdf.date))
    dates.sort()
    matched_close_second = []
    for date in dates:
        bti = btdf[btdf.date == date]
        close_index = np.searchsorted(bti.second, bti.close_second)
        close_index[close_index == len(close_index)] = len(close_index) - 1
        matched_close_second_i = bti.second.values[close_index].tolist()
        matched_close_second.extend(matched_close_second_i)
    return matched_close_second


def dynamic_hold(bti, config, i):
    mid_change = bti.mid - bti.mid[i]
    cond = (mid_change >= config['unwinding_tick_move_upper_bound']) |\
           (mid_change <= config['unwinding_tick_move_lower_bound']) & (mid_change.index > i)
    idx = cond.index[cond]
    idx = idx[0] if len(idx) > 0 else len(idx)-1
    return idx


def get_dynamic_period_close_second(btdf, config):
    dates = list(set(btdf.date))
    dates.sort()
    matched_close_second = []
    for date in dates:
        logger.debug('Getting dynamic holding end time on %s', date)
        bti = btdf[btdf.date == date]
        close_index = [np.nan if bti.trade[i] == 0 else dynamic_hold(bti, config, i) for i in bti.index]
        matched_close_second_i = bti.second[close_index].tolist()
        matched_close_second.extend(matched_close_second_i)
    return matched_close_second


def pnl(btdf, config):
    logger.info('Computing PnL...')
    if config['use_mid']:
        btdf['open_price'] = btdf.mid
    else:
        btdf['open_price'] = (btdf.trade > 0) * btdf.s1 + (btdf.trade < 0) * btdf.b1
    if config['dynamic_unwinding']:
        btdf['matched_close_second'] = get_dynamic_period_close_second(btdf, config)
    else:
        btdf['matched_close_second'] = get_fixed_period_close_second(btdf, config)
    dummy_bt = btdf[['date', 'second', 'b1', 's1', 'mid']].copy()
    dummy_bt.columns = ['date', 'matched_close_second', 'close_b1', 'close_s1', 'close_mid']
    btdf = utils.left_join(btdf, dummy_bt, ['date', 'matched_close_second'])
    if config['use_mid']:
        btdf['close_price'] = btdf.close_mid
    else:
        btdf['close_price'] = (btdf.trade > 0) * btdf.close_b1 + (btdf.trade < 0) * btdf.close_s1
    btdf['pnl'] = btdf.trade * (btdf.close_price - btdf.open_price)
    btdf['transaction_fee'] = config['transaction_fee'] * np.abs(btdf.trade) * (btdf.open_price + btdf.close_price)
    btdf['net_pnl'] = btdf['pnl'] - btdf['transaction_fee']
    logger.info('Finished PnL calculation')
    return btdf


def save(btdf, config):
    file_path = os.path.join(config['data_path'], 'backtest', config['name'])
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    bt_file = os.path.join(file_path, 'backtest.pkl')
    logger.info('Saving backtesting result to %s', bt_file)
    btdf.to_pickle(bt_file)
    config_file = os.path.join(file_path, 'config.pkl')
    logger.info('Saving config file to %s', config_file)
    with open(config_file, 'wb') as cf:
        pickle.dump(config, cf)
    return


def daily_summary(btdf):
    trades = btdf[btdf.trade != 0]
    f = {'pnl': 'sum', 'transaction_fee': 'sum', 'net_pnl': 'sum'}
    daily = trades.groupby('date').agg(f)
    daily['n_trades'] = trades.groupby('date').size()
    return daily


def summary(btdf, config):
    trades = btdf[btdf.trade != 0]
    res = dict()
    res['training_period'] = config['training_period']
    res['trade_trigger_threshold'] = config['trade_trigger_threshold'][1]
    res['holding_period'] = config['holding_period']
    res['use_mid'] = config['use_mid']

    res['n_trades'] = trades.shape[0]
    res['n_trading_days'] = len(set(trades.date))
    res['n_trades_per_day'] = utils.safe_divide(res['n_trades'], res['n_trading_days'])

    res['total_pnl'] = trades.pnl.sum()
    res['total_net_pnl'] = trades.net_pnl.sum()

    res['pnl_per_trade'] = trades.pnl.mean()
    res['net_pnl_per_trade'] = trades.net_pnl.mean()

    res['net_pnl_per_day'] = utils.safe_divide(res['total_net_pnl'], res['n_trading_days'])
    res['pnl_per_day'] = utils.safe_divide(res['total_pnl'], res['n_trading_days'])

    res['std_pnl_per_trade'] = trades.pnl.std()
    res['std_net_pnl_per_trade'] = trades.net_pnl.std()
    return pd.Series(res, name='value')
