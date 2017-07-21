import os
import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

# load data
# ---------

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

# daily statistics
# ----------------

df = pd.DataFrame()
df['date'] = dates
df['win'] = np.repeat(np.nan, n_dates)
df['draw'] = np.repeat(np.nan, n_dates)
df['lose'] = np.repeat(np.nan, n_dates)
df['pnl'] = np.repeat(np.nan, n_dates)
df['rv'] = np.repeat(np.nan, n_dates)
df['hml'] = np.repeat(np.nan, n_dates)
df['break1'] = np.repeat(np.nan, n_dates)
df['break2'] = np.repeat(np.nan, n_dates)

for date in dates:
    dailyPx = px.loc[px.date == date, 'mid']
    seconds = px.loc[px.date == date, 'second']
    mid_diff = (dailyPx - dailyPx.shift(1)).values / tick_size
    mid_diff = mid_diff[1:]
    win = np.sum(mid_diff > 0) / len(mid_diff)
    lose = np.sum(mid_diff < 0) / len(mid_diff)
    draw = np.sum(mid_diff == 0) / len(mid_diff)
    pnl = np.sum(mid_diff)
    hml = (dailyPx.max() - dailyPx.min()) / tick_size
    df.loc[df.date == date, 'win'] = win
    df.loc[df.date == date, 'draw'] = draw
    df.loc[df.date == date, 'lose'] = lose
    df.loc[df.date == date, 'pnl'] = pnl
    df.loc[df.date == date, 'rv'] = np.sqrt(np.nansum(mid_diff * mid_diff) / len(mid_diff))
    df.loc[df.date == date, 'hml'] = hml

    break1 = dailyPx[seconds >= 5400].values[0] - dailyPx[seconds <= 4500].values[-1]
    break2 = dailyPx[seconds >= 16200].values[0] - dailyPx[seconds <= 9000].values[-1]
    df.loc[df.date == date, 'break1'] = break1 / tick_size
    df.loc[df.date == date, 'break2'] = break2 / tick_size

df['format_date'] = format_dates
df.set_index('format_date', inplace=True)

df[['win', 'draw', 'lose']].plot()
df[['win', 'lose']].plot()
(df.win - df.lose).plot()
df.pnl.plot()
df.rv.plot()
df.hml.plot()
df.break1.hist()
df.break2.hist()
