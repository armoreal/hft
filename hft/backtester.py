"""
Backtest Strategy
"""

import logging
import pandas as pd

import hft.utils as utils
import hft.signal_utils as signal

logger = logging.getLogger(__name__)


def select_feature(px, config):
    """Select features to fit model

    :param px: pandas data frame
    :param config: dictionary, config parameters
    :return: list of strings, column names
    """
    y_name = utils.get_moving_column_name(config['response_column'], 0, config['holding_period'])
    selected_features = []
    for feature in config['feature_column']:
        logger.debug('Computing correlation of %s and %s', feature, config['response_column'])
        winsorize_option = {'x_prob': config['feature_winsorize_prob'][feature],
                            'x_bound': config['feature_winsorize_bound'][feature],
                            'y_prob': config['response_winsorize_prob'],
                            'y_bound': config['response_winsorize_bound']
                            }
        corr_mat = signal.xy_corr(px, config['feature_freq'], feature, config['response_column'], winsorize_option)
        correlation = corr_mat.loc[y_name]
        selected_features.append(correlation.argmax())
    return selected_features


def fit(px, features, config):
    """Fit linear model using features

    :param px: pandas data frame, must contain columns in features
    :param features: list of column names
    :param config: dictionary, config parameters
    :return: sklearn model class
    """
    pass


def predict(px, features, model):
    """Predict based on estimated model

    :param px: pandas data frame, must contain columns in features
    :param features: list of column names
    :param model: pandas data frame, coefficients of linear model
    :return: array like predicted values
    """
    pass


def backtest(px, config):
    dates = list(set(px.date))
    dates.sort()
    y_name = utils.get_moving_column_name(config['response_column'], 0, config['holding_period'])
    bt = pd.DataFrame()
    for i in range(config['training_period'], len(dates)):
        date = dates[i]
        logger.info('Backtesting on %s', date)
        logger.info('Selecting feature')
        train = px[(px.date >= dates[i-config['training_period']]) & (px.date < date)].copy()
        features = select_feature(train, config)
        logger.info('Fitting model')
        model = fit(train, features, config)
        logger.info('Predicting future return')
        px_i = px.loc[date, features + [y_name]].copy()
        alpha = predict(px_i, features, model)
        px_i['alpha'] = alpha
        logger.info('Making trading decision')
        px_i['trade'] = 0
        px_i.loc[px_i.alpha > config['trade_trigger_threshold'][1]] = 1
        px_i.loc[px_i.alpha < config['trade_trigger_threshold'][0]] = -1
        bt = bt.append(px_i)
    return bt


