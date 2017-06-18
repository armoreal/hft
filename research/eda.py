import numpy as np
import matplotlib.pylab as plt
import logging

import hft.data_loader as dl
import hft.utils as utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')
product = 'cu'  # switch between cu and zn

# some random day
# ---------------

yyyymmdd = '20131226'
px = dl.load_active_contract(product, yyyymmdd)
px.price.plot()
px[['price', 'mid']].plot()

# overall eda
# -----------

dates = dl.get_dates()
pxall = dl.load_active_contract_multiple_dates(product, dates)

# daily aggregate
# we are more interested in intraday behavior

daily_funs = {'price': 'last', 'volume': 'last', 'outstanding_change': 'sum', 'spread': 'mean', 'mid': 'mean'}
daily_px = pxall.groupby('date').agg(daily_funs)
daily_px.volume.plot(title='volume')
daily_px.price.plot(title='close')
daily_px.spread.plot(title='avg spread')
daily_px.mid.plot(title='avg mid px')
daily_px[['price', 'mid']].plot()
plt.plot(daily_px.price, daily_px.volume, 'o')
plt.plot(daily_px.volume, daily_px.spread, 'o')

# intraday

pxall['hour'] = pxall.index.hour
pxall['minute'] = pxall.index.minute + 60 * pxall.index.hour
funs = {'mid': np.mean, 'qty': np.sum, 'spread': np.mean, 'outstanding_change': np.sum,
        'return': lambda x: np.nansum(x*x)}
rename_dict = {'return': 'realized_vol'}

hourly_px = utils.aggregate(pxall, 'hour', funs, rename_dict)

minutely_px = utils.aggregate(pxall, 'minute', funs, rename_dict)
minutely_px['n_trades'].plot(title='# trades')
minutely_px['qty'].plot(title='volume')
minutely_px['spread'].plot(title='spread')
minutely_px['realized_vol'].plot(title='realized volatility')
plt.plot(minutely_px.spread, minutely_px.realized_vol, 'o')
