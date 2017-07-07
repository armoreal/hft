"""
Backtest Strategy
"""

import logging
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
        px_i['trade'] = 0
        px_i.loc[px_i.alpha > config['trade_trigger_threshold'][1]] = 1
        px_i.loc[px_i.alpha < config['trade_trigger_threshold'][0]] = -1
        bt = bt.append(px_i)
    return bt
