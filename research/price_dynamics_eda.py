import os
import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')


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

# daily high - low
# ----------------

df = pd.DataFrame()
df['date'] = dates
df['hml'] = np.repeat(np.nan, n_dates)
for date in dates:
    dailyPx = px.loc[px.date == date, 'mid']
    hml = (dailyPx.max() - dailyPx.min()) / tick_size
    df.loc[df.date == date, 'hml'] = hml

df['format_date'] = format_dates
df.set_index('format_date', inplace=True)
df.hml.plot()

# daily winning rate
# ------------------

df = pd.DataFrame()
df['date'] = dates
df['win'] = np.repeat(np.nan, n_dates)
df['draw'] = np.repeat(np.nan, n_dates)
df['lose'] = np.repeat(np.nan, n_dates)
for date in dates:
    dailyPx = px.loc[px.date == date, 'mid']
    mid_diff = (dailyPx - dailyPx.shift(1)).values / tick_size
    mid_diff = mid_diff[1:]
    win = np.sum(mid_diff > 0) / len(mid_diff)
    lose = np.sum(mid_diff < 0) / len(mid_diff)
    draw = np.sum(mid_diff == 0) / len(mid_diff)
    df.loc[df.date == date, 'win'] = win
    df.loc[df.date == date, 'draw'] = draw
    df.loc[df.date == date, 'lose'] = lose

df['format_date'] = format_dates
df.set_index('format_date', inplace=True)
df[['win', 'draw', 'lose']].plot()
