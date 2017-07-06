import os
import json
import logging
import numpy as np
import pandas as pd


import hft.utils as utils
import hft.signal_utils as signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

hft_path = os.path.join(os.environ['HOME'], 'dropbox', 'hft')
data_path = os.path.join(hft_path, 'data')
research_path = os.path.join(hft_path, 'research')

# load enriched data
# ------------------

product = 'zn'  # switch between cu and zn
with open(os.path.join(data_path, 'ticksize.json')) as ticksize_file:
    ticksize_json = json.load(ticksize_file)
tick_size = ticksize_json[product]

px = pd.read_pickle(os.path.join(data_path, product+'_enriched.pkl'))

px20131031 = px[px.date == '2013-10-31']
# px = px[np.isnan(px.tick_move_1_0) | (np.abs(px.tick_move_1_0) <= 5)]
px = px[px.date != '2013-10-31']

# signal and return distribution
# ------------------------------

px[['order_flow_imbalance_1_0', 'order_flow_imbalance_2_0', 'order_flow_imbalance_5_0', 'order_flow_imbalance_10_0',
    'order_flow_imbalance_20_0', 'order_flow_imbalance_30_0', 'order_flow_imbalance_60_0',
    'order_flow_imbalance_120_0', 'order_flow_imbalance_180_0', 'order_flow_imbalance_300_0']].describe()

px[['order_imbalance_ratio_1_0', 'order_imbalance_ratio_2_0', 'order_imbalance_ratio_5_0', 'order_imbalance_ratio_10_0',
    'order_imbalance_ratio_20_0', 'order_imbalance_ratio_30_0', 'order_imbalance_ratio_60_0',
    'order_imbalance_ratio_120_0', 'order_imbalance_ratio_180_0', 'order_imbalance_ratio_300_0']].describe()

px[['tick_move_1_0', 'tick_move_2_0', 'tick_move_5_0', 'tick_move_10_0', 'tick_move_20_0', 'tick_move_30_0',
    'tick_move_60_0', 'tick_move_120_0', 'tick_move_180_0', 'tick_move_300_0']][px.date != '2013-10-31'].describe()

px[['tick_move_5_0', 'tick_move_0_10', 'tick_move_0_20']].describe()

signal.plot_two_hist(px, 'order_flow_imbalance', 60, 300)
signal.plot_two_hist(px, 'order_imbalance_ratio', 60, 300)
signal.plot_two_hist(px, 'tick_move', 60, 300)

px.groupby(np.abs(px.tick_move_0_10)).size()

print(sum(px.tick_move_0_10 == 0) / sum(~np.isnan(px.tick_move_0_10)))  # % no move
print(sum(np.abs(px.tick_move_0_10) >= 1) / sum(~np.isnan(px.tick_move_0_10)))  # % 1 tick move
print(sum(np.abs(px.tick_move_0_10) >= 2) / sum(~np.isnan(px.tick_move_0_10)))  # % 2 tick move

print(sum(px.tick_move_0_20 == 0) / sum(~np.isnan(px.tick_move_0_20)))  # % no move
print(sum(np.abs(px.tick_move_0_20) >= 1) / sum(~np.isnan(px.tick_move_0_20)))  # % 1 tick move
print(sum(np.abs(px.tick_move_0_20) >= 2) / sum(~np.isnan(px.tick_move_0_20)))  # % 2 tick move

# scatter plot
# ------------

# forward return by signal
signal.plot_two_scatter(px, 'order_imbalance_ratio', 'tick_move', 1, 0, 0, 1, 5, 0, 0, 5)
signal.plot_two_scatter(px, 'order_flow_imbalance', 'tick_move', 60, 0, 0, 60, 300, 0, 0, 300)
signal.plot_two_scatter(px, 'tick_move',  'tick_move', 5, 0, 0, 5, 60, 0, 0, 60)

# signal by signal
signal.plot_two_scatter(px, 'order_imbalance_ratio', 'tick_move', 1, 0, 1, 0, 5, 0, 5, 0)
signal.plot_two_scatter(px, 'order_flow_imbalance', 'tick_move', 60, 0, 60, 0, 300, 0, 300, 0)
signal.plot_two_scatter(px, 'order_flow_imbalance', 'order_imbalance_ratio', 60, 0, 60, 0, 300, 0, 300, 0)

# correlations
# ------------


second_list = [1, 2, 5, 10, 20, 30, 60, 120, 180, 300]
for sec in second_list:
    px = px[(px[utils.get_moving_column_name('tick_move', 0, sec)] <= 10) | np.isnan(px.tick_move_1_0)]
    px = px[(px[utils.get_moving_column_name('tick_move', sec, 0)] <= 10) | np.isnan(px.tick_move_1_0)]

oir_corr = signal.xy_corr(px, second_list, 'order_imbalance_ratio')
ofi_corr = signal.xy_corr(px, second_list, 'order_flow_imbalance')
autocorr = signal.xy_corr(px, second_list, 'tick_move')
oir_corr.to_csv(os.path.join(research_path, 'oir_corr.csv'))
ofi_corr.to_csv(os.path.join(research_path, 'ofi_corr.csv'))
autocorr.to_csv(os.path.join(research_path, 'autocorr.csv'))

oir_ofi = signal.xx_corr(px, second_list, 'order_imbalance_ratio', 'order_flow_imbalance')
oir_return = signal.xx_corr(px, second_list, 'order_imbalance_ratio', 'tick_move')
ofi_return = signal.xx_corr(px, second_list, 'order_flow_imbalance', 'tick_move')
oir_ofi.to_csv(os.path.join(research_path, 'oir_ofi_corr.csv'))
oir_return.to_csv(os.path.join(research_path, 'oir_return_corr.csv'))
ofi_return.to_csv(os.path.join(research_path, 'ofi_return_corr.csv'))

# multivariate regression
# -----------------------

freq_oir = 1
freq_ofi = 5
freq_xreturn = 2
freq_yreturn = 10

res = signal.reg(px, freq_oir, freq_ofi, freq_xreturn, freq_yreturn, True)

