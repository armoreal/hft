import os
import json
import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

hft_path = os.path.join(os.environ['HOME'], 'dropbox', 'hft')
data_path = os.path.join(hft_path, 'data')
research_path = os.path.join(hft_path, 'research')

# load enriched data
# ------------------

product = 'cu'  # switch between cu and zn
with open(os.path.join(data_path, 'ticksize.json')) as ticksize_file:
    ticksize_json = json.load(ticksize_file)
px = pd.read_pickle(os.path.join(data_path, product+'.pkl'))

if product == 'zn':
    # remove unusual day for zn
    px20131031 = px[px.date == '2013-10-31']
    px = px[px.date != '2013-10-31']

dates = list(set(px.date.tolist()))
dates.sort()
n_dates = len(dates)
format_dates = [datetime.strptime(x, '%Y-%m-%d') for x in dates]

# test hmm!
# --------

date = '2013-10-09'
dailyPx = px[['date', 'mid']][px.date == date]
dailyPx['tick_move'] = (dailyPx['mid']-dailyPx['mid'].shift(1)) / ticksize_json[product]
dailyPx.mid.plot()
dailyPx.tick_move.plot()

x = dailyPx[dailyPx.date == date].tick_move.values[1:]
x[np.abs(x) > 3] = 3
x = x.reshape(x.size, 1)
model = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=50)
model.fit(x)

# hmm parameter by date
# ---------------------

n_comp = 3
transmat = np.repeat(np.nan, n_comp*n_comp*n_dates).reshape(n_dates, n_comp, n_comp)
emission_mean = np.repeat(np.nan, n_comp*n_dates).reshape(n_dates, n_comp)
emission_std = np.repeat(np.nan, n_comp*n_dates).reshape(n_dates, n_comp)
starting_prob = np.repeat(np.nan, n_comp*n_dates).reshape(n_dates, n_comp)

for i, date in enumerate(dates):
    print('Fit HMM on ' + date)
    dailyPx = px[['date', 'mid']][px.date == date]
    dailyPx['tick_move'] = (dailyPx['mid'] - dailyPx['mid'].shift(1)) / ticksize_json[product]
    x = dailyPx[dailyPx.date == date].tick_move.values[1:]
    x[np.abs(x) > 3] = 3
    x = x.reshape(x.size, 1)
    model = hmm.GaussianHMM(n_components=n_comp, n_iter=50)
    model.fit(x)
    index = np.argsort(model.means_.reshape(n_comp))  # sort states based on means
    transmat[i, :, :] = model.transmat_[index, index]
    emission_mean[i, :] = model.means_.reshape(n_comp)[index]
    emission_std[i, :] = np.sqrt(model.covars_).reshape(n_comp)[index]
    starting_prob[i, :] = model.startprob_[index]

plt.plot(format_dates, emission_mean[:, 0], 'r')
plt.plot(format_dates, emission_mean[:, 1], 'b')
plt.plot(format_dates, emission_mean[:, 2], 'g')
plt.show()

plt.plot(format_dates, emission_std[:, 0], 'r')
plt.plot(format_dates, emission_std[:, 1], 'b')
plt.plot(format_dates, emission_std[:, 2], 'g')
plt.show()

plt.plot(format_dates, starting_prob[:, 0], 'r')
plt.plot(format_dates, starting_prob[:, 1], 'b')
plt.plot(format_dates, starting_prob[:, 2], 'g')
plt.show()

plt.plot(format_dates, transmat[:, 0, 0], 'r')
plt.plot(format_dates, transmat[:, 0, 1], 'b')
plt.plot(format_dates, transmat[:, 0, 2], 'g')
plt.show()

plt.plot(format_dates, transmat[:, 1, 0], 'r')
plt.plot(format_dates, transmat[:, 1, 1], 'b')
plt.plot(format_dates, transmat[:, 1, 2], 'g')
plt.show()

plt.plot(format_dates, transmat[:, 2, 0], 'r')
plt.plot(format_dates, transmat[:, 2, 1], 'b')
plt.plot(format_dates, transmat[:, 2, 2], 'g')
plt.show()
