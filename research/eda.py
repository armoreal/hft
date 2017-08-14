import logging
import numpy as np
import matplotlib.pylab as plt

import hft.data_loader as dl
import hft.utils as utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

# some random day
# ---------------

product = 'cu'  # switch between cu and zn
yyyymmdd = '20131015'
px = dl.load_active_contract(product, yyyymmdd)
px.price.plot()
px[['price', 'mid']].plot()

# overall eda
# -----------

dates = dl.get_dates()
pxall = dl.load_active_contract_multiple_dates(product, dates)

# daily aggregate
# we are more interested in intraday behavior

daily_funs = {'price': 'last', 'volume': 'last', 'open_interest': 'sum', 'spread': 'mean', 'mid': 'mean',
              'return': lambda x: np.nansum(x * x)}
daily_px = pxall.groupby('date').agg(daily_funs)
daily_px.rename(columns={'return': 'realized_vol'}, inplace=True)
daily_px.volume.plot(title='volume')
daily_px.price.plot(title='close')
daily_px.spread.plot(title='avg spread')
daily_px.mid.plot(title='avg mid px')
daily_px.realized_vol.plot(title='realized volatility')
daily_px[['price', 'mid']].plot()
plt.plot(daily_px.price, daily_px.volume, 'o')
plt.plot(daily_px.volume, daily_px.spread, 'o')

# intraday

pxall['hour'] = pxall.index.hour
pxall['minute'] = pxall.index.minute + 60 * pxall.index.hour
funs = {'mid': np.mean, 'qty': np.sum, 'spread': np.mean, 'open_interest': np.sum,
        'b1_size': np.mean, 's1_size': np.mean, 'return': lambda x: np.nansum(x*x)}
rename_dict = {'return': 'realized_vol'}

hourly_px = utils.aggregate(pxall, 'hour', funs, rename_dict)

minutely_px = utils.aggregate(pxall, 'minute', funs, rename_dict)
minutely_px['n_trades'].plot(title='# trades')
minutely_px['qty'].plot(title='volume')
minutely_px['spread'].plot(title='spread')
minutely_px['realized_vol'].plot(title='realized volatility')
minutely_px[['b1_size', 's1_size']].plot()
(minutely_px.b1_size - minutely_px.s1_size).plot(title='b1_size - s1_size')
plt.plot(minutely_px.spread, minutely_px.realized_vol, 'o')
