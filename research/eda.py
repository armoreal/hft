import os
import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
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

daily_funs = {'price': 'last', 'volume': 'last', 'open_interest': 'sum', 'spread': 'mean', 'mid': 'mean'}
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

# adopt new data loading
# ----------------------

hft_path = os.path.join(os.environ['HOME'], 'dropbox', 'hft')
data_path = os.path.join(hft_path, 'data')
research_path = os.path.join(hft_path, 'research')

product = 'cu'  # switch between cu and zn
with open(os.path.join(data_path, 'ticksize.json')) as ticksize_file:
    ticksize_json = json.load(ticksize_file)
tick_size = ticksize_json[product]
px = pd.read_pickle(os.path.join(data_path, product+'.pkl'))

if product == 'zn':
    # remove unusual day for zn
    px20131031 = px[px.date == '2013-10-31']
    px = px[px.date != '2013-10-31']

dates = list(set(px.date.tolist()))
dates.sort()
n_dates = len(dates)
format_dates = [datetime.strptime(x, '%Y-%m-%d') for x in dates]

# break tick move
# ---------------

break_df = pd.DataFrame()
break_df['date'] = dates
break_df['break1'] = np.repeat(np.nan, n_dates)
break_df['break2'] = np.repeat(np.nan, n_dates)
for date in dates:
    dailyPx = px[px.date == date]
    break1 = dailyPx.loc[dailyPx.second >= 5400, 'mid'].values[0] - \
        dailyPx.loc[dailyPx.second <= 4500, 'mid'].values[-1]
    break2 = dailyPx.loc[dailyPx.second >= 16200, 'mid'].values[0] - \
        dailyPx.loc[dailyPx.second <= 9000, 'mid'].values[-1]
    break_df.loc[break_df.date == date, 'break1'] = break1 / tick_size
    break_df.loc[break_df.date == date, 'break2'] = break2 / tick_size

break_df.break1.hist()
break_df.break2.hist()

# daily realized volatility
# -------------------------

rvdf = pd.DataFrame()
rvdf['date'] = dates
rvdf['rv'] = np.repeat(np.nan, n_dates)
for date in dates:
    dailyPx = px.loc[px.date == date, 'mid']
    mid_diff = (dailyPx - dailyPx.shift(1)).values / tick_size
    rvdf.loc[rvdf.date == date, 'rv'] = np.sqrt(np.nansum(mid_diff*mid_diff) / len(mid_diff))

rvdf['format_date'] = format_dates
rvdf.set_index('format_date', inplace=True)
rvdf.rv.plot()
