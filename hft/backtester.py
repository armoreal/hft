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
    dates = list(set(px.date))
    dates.sort()
    y_name = utils.get_moving_column_name(config['response_column'], 0, config['holding_period'])
    bt = pd.DataFrame()
    columns = ['dt', 'date', 'time', 'price', 'qty', 'volume', 'open_interest',
               'b1', 'b1_size', 's1', 's1_size', 'mid', 'second']
    for i in range(config['training_period'], len(dates)):
        date = dates[i]
        logger.info('Backtesting on %s', date)
        logger.info('Selecting feature')
        train = px[(px.date >= dates[i-config['training_period']]) & (px.date < date)].copy()
        features = select_feature(train, config)
        logger.info('Fitting model')
        model = fit(train, features, config)
        logger.info('Predicting future return')
        px_i = px.loc[px.date == date, columns + features + [y_name]].copy()
        x_new = px_i[features]
        x_new = x_new.fillna(x_new.median())
        alpha = model.predict(x_new)
        px_i['alpha'] = alpha
        logger.info('Making trading decision')
        bt = bt.append(px_i)
    logger.info('Finish backtesting')
    return bt


def trade(bt, config):
    bt['trade'] = 0
    bt.loc[bt.alpha > config['trade_trigger_threshold'][1], 'trade'] = 1
    bt.loc[bt.alpha < config['trade_trigger_threshold'][0], 'trade'] = -1
    bt.loc[bt.second > config['end_second'], 'trade'] = 0
    bt.loc[bt.second < config['start_second'], 'trade'] = 0
    return bt


def get_close_second(bt, config):
    bt['close_second'] = bt.second + config['holding_period']
    dates = list(set(bt.date))
    dates.sort()
    matched_close_second = []
    for date in dates:
        bti = bt[bt.date == date]
        close_index = np.searchsorted(bti.second, bti.close_second)
        close_index[close_index == len(close_index)] = len(close_index) - 1
        matched_close_second_i = bti.second.values[close_index].tolist()
        matched_close_second.extend(matched_close_second_i)
    return matched_close_second


def pnl(bt, config):
    logger.info('Computing PnL...')
    if config['use_mid']:
        bt['open_price'] = bt.mid
    else:
        bt['open_price'] = (bt.trade > 0) * bt.s1 + (bt.trade < 0) * bt.b1
    bt['matched_close_second'] = get_close_second(bt, config)
    dummy_bt = bt[['date', 'second', 'b1', 's1', 'mid']].copy()
    dummy_bt.columns = ['date', 'matched_close_second', 'close_b1', 'close_s1', 'close_mid']
    bt = pd.merge(bt, dummy_bt, on=['date', 'matched_close_second'], how='left')
    if config['use_mid']:
        bt['close_price'] = bt.close_mid
    else:
        bt['close_price'] = (bt.trade > 0) * bt.close_b1 + (bt.trade < 0) * bt.close_s1
    bt['pnl'] = bt.trade * (bt.close_price - bt.open_price)
    bt['transaction_fee'] = config['transaction_fee'] * np.abs(bt.trade) * (bt.open_price + bt.close_price)
    logger.info('Finished PnL calculation')
    return bt


def save(bt, config):
    file_path = os.path.join(config['data_path'], 'backtest', config['name'])
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    bt_file = os.path.join(file_path, 'backtest.pkl')
    logger.info('Saving backtesting result to %s', bt_file)
    bt.to_pickle(bt_file)
    config_file = os.path.join(file_path, 'config.pkl')
    logger.info('Saving config file to %s', config_file)
    with open(config_file, 'wb') as cf:
        pickle.dump(config, cf)
    return
