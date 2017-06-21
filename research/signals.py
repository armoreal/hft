import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

import hft.data_loader as dl
import hft.signal_utils as signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

product = 'cu'  # switch between cu and zn
seconds = 10
return_col = 'return'+str(seconds)
return_cutoff = 6

# load raw data
dates = dl.get_dates()
pxall = dl.load_active_contract_multiple_dates(product, dates)

# get signals on all dates
px = pxall.copy()
px = signal.signal_on_multiple_dates(px, signal.order_imbalance_ratio)
px = signal.signal_on_multiple_dates(px, signal.volume_order_imbalance)
px = signal.signal_on_multiple_dates(px, lambda x: signal.period_return(x, seconds))
px['bps_return'] = px[return_col] * 1e4

# data cleaning
# -------------

px[['volume_order_imbalance', 'order_imbalance_ratio', 'bps_return']].describe()

# ~50% of return does not change within 10 seconds
print(sum(px.bps_return == 0) / sum(~np.isnan(px.bps_return)))

px.loc[np.abs(px.bps_return) > return_cutoff, 'bps_return'] =\
    np.sign(px.loc[np.abs(px.bps_return) > return_cutoff, 'bps_return']) * return_cutoff
px.bps_return.hist(bins=20)

plt.scatter(px.order_imbalance_ratio, px.bps_return, marker='o', s=0.1)

plt.scatter(px.volume_order_imbalance, px.bps_return, marker='o', s=0.1)