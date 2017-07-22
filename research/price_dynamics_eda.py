import os
import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as sm
from sklearn import linear_model

import hft.utils as utils

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
df['mid_acf_1'] = np.repeat(np.nan, n_dates)
df['mid_acf_2'] = np.repeat(np.nan, n_dates)
df['mid_pacf_1'] = np.repeat(np.nan, n_dates)
df['mid_pacf_2'] = np.repeat(np.nan, n_dates)
df['diff_acf_1'] = np.repeat(np.nan, n_dates)
df['diff_acf_2'] = np.repeat(np.nan, n_dates)
df['diff_pacf_1'] = np.repeat(np.nan, n_dates)
df['diff_pacf_2'] = np.repeat(np.nan, n_dates)

for date in dates:
    print('Compute statistics on date ' + date)

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

    mid_acf = sm.acf(dailyPx.values, nlags=2)
    mid_pacf = sm.pacf(dailyPx.values, nlags=2)
    diff_acf = sm.acf(mid_diff, nlags=2)
    diff_pacf = sm.pacf(mid_diff, nlags=2)
    df.loc[df.date == date, 'mid_acf_1'] = mid_acf[1]
    df.loc[df.date == date, 'mid_acf_2'] = mid_acf[2]
    df.loc[df.date == date, 'mid_pacf_1'] = mid_pacf[1]
    df.loc[df.date == date, 'mid_pacf_2'] = mid_pacf[2]
    df.loc[df.date == date, 'diff_acf_1'] = diff_acf[1]
    df.loc[df.date == date, 'diff_acf_2'] = diff_acf[2]
    df.loc[df.date == date, 'diff_pacf_1'] = diff_pacf[1]
    df.loc[df.date == date, 'diff_pacf_2'] = diff_pacf[2]

df['format_date'] = format_dates
df.set_index('format_date', inplace=True)

df[['win', 'draw', 'lose']].plot()
df[['win', 'lose']].plot()
(df.win - df.lose).plot()
df.pnl.plot()
df[['pnl', 'hml']].plot()
df.rv.plot()
df.hml.plot()
df.break1.hist()
df.break2.hist()

df.plot.scatter(x='win', y='lose')
df.plot.scatter(x='win', y='pnl')
df.plot.scatter(x='hml', y='lose')

# ACF and PACF of mid and mid_diff
df[['mid_acf_1', 'mid_acf_2', 'mid_pacf_1', 'mid_pacf_2']].plot()
df[['diff_acf_1', 'diff_acf_2', 'diff_pacf_1', 'diff_pacf_2']].plot()


# same statistics different sample freq
# -------------------------------------

period = 60
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
df['mid_acf_1'] = np.repeat(np.nan, n_dates)
df['mid_acf_2'] = np.repeat(np.nan, n_dates)
df['mid_pacf_1'] = np.repeat(np.nan, n_dates)
df['mid_pacf_2'] = np.repeat(np.nan, n_dates)
df['diff_acf_1'] = np.repeat(np.nan, n_dates)
df['diff_acf_2'] = np.repeat(np.nan, n_dates)
df['diff_pacf_1'] = np.repeat(np.nan, n_dates)
df['diff_pacf_2'] = np.repeat(np.nan, n_dates)

for date in dates:
    print('Compute statistics on date ' + date)

    dailyPx = px[px.date == date]
    dailyPx = utils.get_period_px(dailyPx, period)
    seconds = dailyPx.second
    prices = dailyPx.mid
    mid_diff = (prices - prices.shift(1)).values / tick_size
    mid_diff = mid_diff[1:]
    win = np.sum(mid_diff > 0) / len(mid_diff)
    lose = np.sum(mid_diff < 0) / len(mid_diff)
    draw = np.sum(mid_diff == 0) / len(mid_diff)
    pnl = np.sum(mid_diff)
    hml = (prices.max() - prices.min()) / tick_size
    df.loc[df.date == date, 'win'] = win
    df.loc[df.date == date, 'draw'] = draw
    df.loc[df.date == date, 'lose'] = lose
    df.loc[df.date == date, 'pnl'] = pnl
    df.loc[df.date == date, 'rv'] = np.sqrt(np.nansum(mid_diff * mid_diff) / len(mid_diff))
    df.loc[df.date == date, 'hml'] = hml

    break1 = prices[seconds >= 5400].values[0] - prices[seconds <= 4500].values[-1]
    break2 = prices[seconds >= 16200].values[0] - prices[seconds <= 9000].values[-1]
    df.loc[df.date == date, 'break1'] = break1 / tick_size
    df.loc[df.date == date, 'break2'] = break2 / tick_size

    mid_acf = sm.acf(prices.values, nlags=2)
    mid_pacf = sm.pacf(prices.values, nlags=2)
    diff_acf = sm.acf(mid_diff, nlags=2)
    diff_pacf = sm.pacf(mid_diff, nlags=2)
    df.loc[df.date == date, 'mid_acf_1'] = mid_acf[1]
    df.loc[df.date == date, 'mid_acf_2'] = mid_acf[2]
    df.loc[df.date == date, 'mid_pacf_1'] = mid_pacf[1]
    df.loc[df.date == date, 'mid_pacf_2'] = mid_pacf[2]
    df.loc[df.date == date, 'diff_acf_1'] = diff_acf[1]
    df.loc[df.date == date, 'diff_acf_2'] = diff_acf[2]
    df.loc[df.date == date, 'diff_pacf_1'] = diff_pacf[1]
    df.loc[df.date == date, 'diff_pacf_2'] = diff_pacf[2]

df['format_date'] = format_dates
df.set_index('format_date', inplace=True)

df[['win', 'draw', 'lose']].plot()
df[['win', 'lose']].plot()
(df.win - df.lose).plot()
df.pnl.plot()
df[['pnl', 'hml']].plot()
df.rv.plot()
df.break1.hist()
df.break2.hist()

df.plot.scatter(x='win', y='lose')
df.plot.scatter(x='win', y='pnl')
df.plot.scatter(x='hml', y='lose')

# ACF and PACF of mid and mid_diff
df[['mid_acf_1', 'mid_acf_2', 'mid_pacf_1', 'mid_pacf_2']].plot()
df[['diff_acf_1', 'diff_acf_2', 'diff_pacf_1', 'diff_pacf_2']].plot()


# fit O-U process
# ---------------

period = 60
df = pd.DataFrame()
df['date'] = dates
df['b0'] = np.repeat(np.nan, n_dates)
df['b1'] = np.repeat(np.nan, n_dates)
df['mse'] = np.repeat(np.nan, n_dates)
df['rsq'] = np.repeat(np.nan, n_dates)
df['s0'] = np.repeat(np.nan, n_dates)
df['s1'] = np.repeat(np.nan, n_dates)
df['t0'] = np.repeat(np.nan, n_dates)
df['t1'] = np.repeat(np.nan, n_dates)

for date in dates:
    print('Fitting O-U process on date ' + date)

    dailyPx = px[px.date == date]
    dailyPx = utils.get_period_px(dailyPx, period)
    seconds = dailyPx.second
    prices = dailyPx.mid.values

    n = len(prices)-1
    y = prices[1:]
    x = np.array([[1] * n, list(prices[:n])]).transpose()
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    df.loc[df.date == date, 'b0'] = regr.coef_[0]
    df.loc[df.date == date, 'b1'] = regr.coef_[1]
    mse = np.sum((regr.predict(x) - y) ** 2) / (n-2)
    df.loc[df.date == date, 'mse'] = mse
    df.loc[df.date == date, 'rsq'] = regr.score(x, y)
    ssq = np.sum((x[:, 1] - np.mean(x[:, 1])) ** 2)
    s1 = np.sqrt(mse/ssq)
    s0 = np.sqrt(mse/ssq*np.mean(x[:, 1]**2))
    df.loc[df.date == date, 's0'] = s0
    df.loc[df.date == date, 's1'] = s1
    df.loc[df.date == date, 't0'] = regr.coef_[0] / s0
    df.loc[df.date == date, 't1'] = regr.coef_[1] / s1

df['format_date'] = format_dates
df.set_index('format_date', inplace=True)

df[['rsq']].plot()
df[['mse']].plot()
df[['b0', 'b1']].plot()
df[['t0', 't1']].plot()
