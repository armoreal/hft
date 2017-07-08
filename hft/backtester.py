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
        alpha = model.predict(x_new)
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


def get_close_second(btdf, config):
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


def pnl(btdf, config):
    logger.info('Computing PnL...')
    if config['use_mid']:
        btdf['open_price'] = btdf.mid
    else:
        btdf['open_price'] = (btdf.trade > 0) * btdf.s1 + (btdf.trade < 0) * btdf.b1
    btdf['matched_close_second'] = get_close_second(btdf, config)
    dummy_bt = btdf[['date', 'second', 'b1', 's1', 'mid']].copy()
    dummy_bt.columns = ['date', 'matched_close_second', 'close_b1', 'close_s1', 'close_mid']
    btdf = pd.merge(btdf, dummy_bt, on=['date', 'matched_close_second'], how='left')
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
