"""
Back test
"""

import os
import json
import logging
import numpy as np
import pandas as pd

import hft.backtester as bt

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

hft_path = os.path.join(os.environ['HOME'], 'dropbox', 'hft')
data_path = os.path.join(hft_path, 'data')
research_path = os.path.join(hft_path, 'research')

# load enriched data
# ------------------

product = 'zn'  # switch between cu and zn
with open(os.path.join(data_path, 'ticksize.json')) as ticksize_file:
    ticksize_json = json.load(ticksize_file)

px = pd.read_pickle(os.path.join(data_path, product+'_enriched.pkl'))
if product == 'zn':
    # remove unusual day for zn
    px20131031 = px[px.date == '2013-10-31']
    px = px[px.date != '2013-10-31']

# configuration
# -------------

config = dict()

# general configuration
config['name'] = product + '_1'
config['data_path'] = data_path
config['start_date'] = '2013-10-05'

# model specifics
config['training_period'] = 1  # days
config['feature_column'] = ['order_imbalance_ratio', 'order_flow_imbalance', 'tick_move']
config['feature_freq'] = [1, 2, 5, 10, 20, 30, 60, 120, 180, 300]
config['feature_winsorize_prob'] = {'order_imbalance_ratio': [0.0, 0.0],
                                    'order_flow_imbalance': [0.005, 0.005],
                                    'tick_move': [0, 0]}
config['feature_winsorize_bound'] = {'order_imbalance_ratio': [-np.inf, np.inf],
                                     'order_flow_imbalance': [-np.inf, np.inf],
                                     'tick_move': [-10, 10]}
config['response_column'] = 'tick_move'
config['response_winsorize_prob'] = [0, 0]
config['response_winsorize_bound'] = [-5, 5]

# open/close/hold condition
config['holding_period'] = 120  # seconds
config['dynamic_unwinding'] = True
config['unwinding_tick_move_upper_bound'] = 3
config['unwinding_tick_move_lower_bound'] = -3
config['trade_trigger_threshold'] = [-1.5, 1.5]
config['start_second'] = 120
config['end_second'] = 21420

# pnl
config['use_mid'] = False  # if False, use touch price
config['transaction_fee'] = 0.0001  # 1 bps transaction fee

# backtesting
# -----------

btdf = bt.backtest(px, config)
btdf = bt.trade(btdf, config)
btdf = bt.pnl(btdf, config)
# bt.save(btdf, config)

trades = btdf[btdf.trade != 0]
bt.summary(btdf, config)
bt.daily_summary(btdf)
trades.pnl.hist(bins=30)

# pnl vs threshold - fixed period
# -------------------------------

training_periods = [1, 5]
holding_periods = [20, 30, 60, 120, 180, 300]
thresholds = [0.5, 1.0, 1.5, 2.0]
file_path = os.path.join(data_path, 'backtest', product + '_by_hldg_thld')
res_table = pd.DataFrame()

for training_period in training_periods:
    print('############################################')
    print('########## training_period = ' + str(training_period) + ' ##########')
    config['training_period'] = training_period
    for use_mid in [True, False]:
        print('############################################')
        print('########## use_mid = ' + str(use_mid) + ' ##########')
        config['use_mid'] = use_mid
        for hldg in holding_periods:
            print('Compute pnl for Holding period = ' + str(hldg))
            by_thld_table = pd.DataFrame()
            config['holding_period'] = hldg
            btdf = bt.backtest(px, config)
            for thld in thresholds:
                config['trade_trigger_threshold'] = [-thld, thld]
                btdf = bt.trade(btdf, config)
                btdf = bt.pnl(btdf, config)
                by_thld_table[str(thld)] = bt.summary(btdf, config)
            by_thld_table = by_thld_table.transpose()
            res_table = res_table.append(by_thld_table)
            # file_name = os.path.join(file_path, product + '_' + str(hldg) + '.csv')
            # by_thld_table.to_csv(file_name)

file_name = os.path.join(file_path, product + '.csv')
res_table.to_csv(file_name, index=False)

# pnl vs threshold - dynamic holding
# ----------------------------------

training_periods = [1, 5]
thresholds = [0.5, 1.0, 1.5]
holding_periods = [30, 60, 120, 300]
unwinding_upper_bounds = [3, 3, 5, 5]
unwinding_lower_bounds = [-3, -2, -5, -3]
file_path = os.path.join(data_path, 'backtest')
res_table = pd.DataFrame()

for training_period in training_periods:
    print('############################################')
    print('########## training_period = ' + str(training_period) + ' ##########')
    config['training_period'] = training_period
    for hldg in holding_periods:
        print('############################################')
        print('########## Holding_period = ' + str(hldg) + ' ##########')
        config['holding_period'] = hldg
        by_thld_table = pd.DataFrame()
        btdf = bt.backtest(px, config)
        for i_unwinding in range(len(unwinding_lower_bounds)):
            print('Unwinding upper bound = ' + str(unwinding_upper_bounds[i_unwinding]))
            config['unwinding_tick_move_upper_bound'] = unwinding_upper_bounds[i_unwinding]
            config['unwinding_tick_move_lower_bound'] = unwinding_lower_bounds[i_unwinding]
            for thld in thresholds:
                config['trade_trigger_threshold'] = [-thld, thld]
                btdf = bt.trade(btdf, config)
                for use_mid in [True, False]:
                    config['use_mid'] = use_mid
                    btdf = bt.pnl(btdf, config)
                    by_thld_table = bt.summary(btdf, config)
                    res_table = res_table.append(by_thld_table, ignore_index=True)
            # file_name = os.path.join(file_path, product + '_' + str(hldg) + '.csv')
            # by_thld_table.to_csv(file_name)

file_name = os.path.join(file_path, product + '_dynamic_holding.csv')
res_table.to_csv(file_name, index=False)

# exam why positive pnl
# ---------------------

config = dict()

# general configuration
config['name'] = product + '_1'
config['data_path'] = data_path
config['start_date'] = '2013-10-05'

# model specifics
config['training_period'] = 1  # days
config['feature_column'] = ['order_imbalance_ratio', 'order_flow_imbalance', 'tick_move']
config['feature_freq'] = [1, 2, 5, 10, 20, 30, 60, 120, 180, 300]
config['feature_winsorize_prob'] = {'order_imbalance_ratio': [0.0, 0.0],
                                    'order_flow_imbalance': [0.005, 0.005],
                                    'tick_move': [0, 0]}
config['feature_winsorize_bound'] = {'order_imbalance_ratio': [-np.inf, np.inf],
                                     'order_flow_imbalance': [-np.inf, np.inf],
                                     'tick_move': [-10, 10]}
config['response_column'] = 'tick_move'
config['response_winsorize_prob'] = [0, 0]
config['response_winsorize_bound'] = [-5, 5]

# open/close/hold condition
config['holding_period'] = 120  # seconds
config['dynamic_unwinding'] = True
config['unwinding_tick_move_upper_bound'] = 3
config['unwinding_tick_move_lower_bound'] = -3
config['trade_trigger_threshold'] = [-1.5, 1.5]
config['start_second'] = 120
config['end_second'] = 21420

# pnl
config['use_mid'] = False  # if False, use touch price
config['transaction_fee'] = 0.0001  # 1 bps transaction fee

# backtesting

btdf = bt.backtest(px, config)
btdf = bt.trade(btdf, config)
btdf = bt.pnl(btdf, config)
trades = btdf[btdf.trade != 0]
bt.summary(btdf, config)
bt.daily_summary(btdf)
t = trades[trades.date == '2013-12-26']
