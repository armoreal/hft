import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import hft.data_loader as dl
import hft.utils as utils
import hft.signal_utils as signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

data_path = os.path.join(os.environ['HOME'], 'hft')

# load raw data
# -------------

product = 'zn'  # switch between cu and zn
with open(os.path.join(os.environ['HOME'], 'hft', 'ticksize.json')) as ticksize_file:
    ticksize_json = json.load(ticksize_file)
tick_size = ticksize_json[product]

# load raw data
# dates = dl.get_dates()
# pxall = dl.load_active_contract_multiple_dates(product, dates)
# pxall.to_pickle(os.path.join(os.environ['HOME'], 'hft', product+'.pkl'))
pxall = pd.read_pickle(os.path.join(os.environ['HOME'], 'hft', product+'.pkl'))

# cache index
# -----------

px = pxall.copy().reset_index()
second_list = [1, 2, 5, 10, 20]

for sec in second_list:
    print('----------- sec = ' + str(sec) + ' -------------')
    backward_index = utils.get_index_multiple_dates(px, sec, 0)
    file_name = os.path.join(data_path, product+'_index_'+str(sec)+'_0.pkl')
    print('Saving backward index to ' + file_name)
    backward_index.to_pickle(file_name)

    forward_index = utils.get_index_multiple_dates(px, 0, sec)
    file_name = os.path.join(data_path, product+'_index_0_'+str(sec)+'.pkl')
    print('Saving forward index to ' + file_name)
    forward_index.to_pickle(file_name)

# compute signal
# --------------

index_folder = os.path.join(data_path, 'index')
# px = pxall[pxall.date.isin(['2013-10-08', '2013-10-09'])].copy().reset_index()
px = pxall.copy().reset_index()
second_list = [1, 2, 5, 10, 20]

for sec in second_list:
    print('----------- sec = ' + str(sec) + ' -------------')

    # backward metrics
    filename = os.path.join(index_folder, product+'_index_'+str(sec)+'_0.pkl')
    backward_index = pd.read_pickle(filename)
    px = signal.signal_on_multiple_dates(px, lambda x: signal.order_imbalance_ratio(x, sec, 0, backward_index))
    px = signal.signal_on_multiple_dates(px, lambda x: signal.order_flow_imbalance(x, sec, 0, backward_index, False))
    px = signal.signal_on_multiple_dates(px, lambda x: signal.order_flow_imbalance(x, sec, 0, backward_index, True))
    px = signal.signal_on_multiple_dates(px, lambda x: signal.period_mid_move(x, sec, 0, tick_size, backward_index))

    # forward metrics
    filename = os.path.join(index_folder, product + '_index_0_' + str(sec) + '.pkl')
    forward_index = pd.read_pickle(filename)
    px = signal.signal_on_multiple_dates(px, lambda x: signal.period_mid_move(x, 0, sec, tick_size, forward_index))

    # save to file
    filename = os.path.join(data_path, product+'_'+str(sec)+'.pkl')
    print('Saving forward index to ' + filename)
    px.to_pickle(filename)

# data cleaning
# -------------

return_cutoff = 6

px[['volume_order_imbalance', 'order_imbalance_ratio', 'price_change', 'bps_return']].describe()

# ~50% of return does not change within 10 seconds
print(sum(px.bps_return == 0) / sum(~np.isnan(px.bps_return)))

px.loc[np.abs(px.bps_return) > return_cutoff, 'bps_return'] =\
    np.sign(px.loc[np.abs(px.bps_return) > return_cutoff, 'bps_return']) * return_cutoff
px.bps_return.hist(bins=20)

plt.scatter(px.order_imbalance_ratio, px.price_tick_move, marker='o', s=0.1)
plt.xlabel('order imbalance ratio')
plt.ylabel(str(seconds)+'-seconds price tick move')

plt.scatter(px.volume_order_imbalance, px.price_tick_move, marker='o', s=0.1)
plt.xlabel('volume order imbalance')
plt.ylabel(str(seconds)+'-seconds price tick move')
