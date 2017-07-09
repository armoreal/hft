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

product = 'cu'  # switch between cu and zn
with open(os.path.join(data_path, 'ticksize.json')) as ticksize_file:
    ticksize_json = json.load(ticksize_file)

px = pd.read_pickle(os.path.join(data_path, product+'_enriched.pkl'))
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
config['training_period'] = 21  # days
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
config['holding_period'] = 10  # seconds
config['trade_trigger_threshold'] = [-0.4, 0.4]
config['start_second'] = 180
config['end_second'] = 21420

# pnl
config['use_mid'] = False  # if False, use touch price
config['transaction_fee'] = 0.0001  # 1 bps transaction fee

# backtesting
# -----------

btdf = bt.backtest(px, config)
btdf = bt.trade(btdf, config)
btdf = bt.pnl(btdf, config)
bt.save(btdf, config)

trades = btdf[btdf.trade != 0]
trades.pnl.hist(bins=30)

# pnl vs threshold

holding_periods = [5, 10, 20, 30, 60, 120, 180, 300]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
file_path = os.path.join(data_path, 'backtest', product + '_by_hldg_thld')

for hldg in holding_periods:
    print('Compute pnl for Holding period = ' + str(hldg))
    by_thld_table = pd.DataFrame()
    config['holding_period'] = hldg
    for thld in thresholds:
        config['trade_trigger_threshold'] = [-thld, thld]
        btdf = bt.trade(btdf, config)
        btdf = bt.pnl(btdf, config)
        by_thld_table[str(thld)] = bt.summary(btdf)
    by_thld_table = by_thld_table.transpose()
    file_name = os.path.join(file_path, product + '_' + str(hldg) + '.csv')
    print('Saving file to ' + file_name)
    by_thld_table.to_csv(file_name)
