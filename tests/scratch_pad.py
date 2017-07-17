
import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

hft_path = os.path.join(os.environ['HOME'], 'dropbox', 'hft')
data_path = os.path.join(hft_path, 'data')
research_path = os.path.join(hft_path, 'research')

# load enriched data
# ------------------

product = 'cu'  # switch between cu and zn
with open(os.path.join(data_path, 'ticksize.json')) as ticksize_file:
    ticksize_json = json.load(ticksize_file)

px = pd.read_pickle(os.path.join(data_path, product+'_enriched.pkl'))
px20131031 = px[px.date == '2013-10-31']
px = px[px.date != '2013-10-31']

dailyPx = px.groupby('date')[['mid']].transform(pd.Series.diff)
dailyPx['tick_move'] = dailyPx['mid'] / ticksize_json[product]
dailyPx['date'] = px['date']
dailyPx[dailyPx.date == '2013-10-08'].tick_move.plot()
