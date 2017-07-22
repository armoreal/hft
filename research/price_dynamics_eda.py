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


# fit daily O-U process
# ---------------------

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
df['kappa'] = np.repeat(np.nan, n_dates)
df['m'] = np.repeat(np.nan, n_dates)
df['sigma'] = np.repeat(np.nan, n_dates)

for date in dates:
    print('Fitting O-U process on date ' + date)

    dailyPx = px[px.date == date]
    dailyPx = utils.get_period_px(dailyPx, period)
    seconds = dailyPx.second
    prices = dailyPx.mid.values

    y = prices[1:]
    n = len(y)
    x = prices[:-1].reshape(n, 1)
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

    b0 = regr.intercept_
    b1 = regr.coef_.item()
    df.loc[df.date == date, 'b0'] = b0
    df.loc[df.date == date, 'b1'] = b1
    mse = np.sum((regr.predict(x) - y) ** 2) / (n-2)
    df.loc[df.date == date, 'mse'] = mse
    df.loc[df.date == date, 'rsq'] = regr.score(x, y)
    ssq = np.sum((x - np.mean(x)) ** 2)
    s1 = np.sqrt(mse / ssq)
    s0 = np.sqrt(mse / ssq * np.mean(x ** 2))
    df.loc[df.date == date, 's0'] = s0
    df.loc[df.date == date, 's1'] = s1
    df.loc[df.date == date, 't0'] = b0 / s0
    df.loc[df.date == date, 't1'] = b1 / s1

    kappa = -np.log(b1) / period
    df.loc[df.date == date, 'kappa'] = kappa
    df.loc[df.date == date, 'm'] = b0 / (1 - b1)
    df.loc[df.date == date, 'sigma'] = np.sqrt(mse * 2 * kappa / (1 - b1**2))

df['format_date'] = format_dates
df.set_index('format_date', inplace=True)

df[['rsq']].plot()
df[['mse']].plot()
df[['b0']].plot()
df[['b1']].plot()
df[['t0', 't1']].plot()

df.kappa.plot()
df.m.plot()
df.sigma.plot()

# aggregate all the tick move and fit OU process
# ----------------------------------------------

period = 60
price_delta = []
for date in dates:
    print('Gathering prices on ' + date)
    dailyPx = px[px.date == date]
    dailyPx = utils.get_period_px(dailyPx, period)
    prices = dailyPx.mid.values
    delta = (prices[1:] - prices[:-1]) / tick_size
    price_delta += list(delta)

price_delta = np.array(price_delta)
prices = np.cumsum(price_delta)
y = prices[1:]
n = len(y)
x = prices[:-1].reshape(n, 1)
regr = linear_model.LinearRegression()
regr.fit(x, y)
b0 = regr.intercept_
b1 = regr.coef_.item()
mse = np.sum((regr.predict(x) - y) ** 2) / (n-2)
rsq = regr.score(x, y)
ssq = np.sum((x - np.mean(x)) ** 2)
s1 = np.sqrt(mse/ssq)
s0 = np.sqrt(mse / ssq * np.mean(x ** 2))
t0 = b0 / s0
t1 = b1 / s1

kappa = -np.log(b1) / period
m = b0 / (1 - b1)
sigma = np.sqrt(mse * 2 * kappa / (1 - b1 ** 2))

