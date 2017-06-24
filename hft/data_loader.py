"""
Data Loading Functions
"""

import os
import logging
import json
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.environ['HOME'], 'hft', 'SpRawFutureTick')
with open(os.path.join('config', 'data_loader.json')) as data_config_file:
    data_config = json.load(data_config_file)
COLUMNS = data_config['columns']
COLUMNS_TO_DROP = data_config['columns_to_drop']
ENCODING = data_config['encoding']


def get_dates():
    return os.listdir(DATA_PATH)


def get_filenames(product, yyyymmdd):
    contract_month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    filenames = [os.path.join(DATA_PATH, yyyymmdd, product + x + '_' + yyyymmdd + '.csv') for x in contract_month]
    filenames = [x for x in filenames if os.path.isfile(x)]
    return filenames


def process_raw_table(px):
    px.columns = COLUMNS
    px['spread'] = px['s1'] - px['b1']
    px['mid'] = 0.5 * (px['b1']+px['s1'])
    px['return'] = (px['mid'] - px['mid'].shift(1)) / px['mid'].shift(1)
    px['dt'] = px['date'] + ' ' + px['time']
    px['dt'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in px['dt']]
    px.set_index('dt', inplace=True)
    px['second'] = (px.index.hour-9)*3600 + px.index.minute*60 + px.index.second
    half_second_index = px.second == px.second.shift(-1)
    px.loc[half_second_index, 'second'] = px.loc[half_second_index, 'second'] - 0.5
    px.drop(COLUMNS_TO_DROP, axis=1, inplace=True)
    return px


def load_contract(product, yyyymmdd, contract_month):
    logger.debug('Loading %s-%s data on %s', product, contract_month, yyyymmdd)
    filename = os.path.join(DATA_PATH, yyyymmdd, product + contract_month + '_' + yyyymmdd + '.csv')
    px = pd.read_csv(filename, encoding=ENCODING)
    px = process_raw_table(px)
    return px


def load_active_contract(product, yyyymmdd):
    logger.debug('Loading %s active contract data on %s', product, yyyymmdd)
    filenames = get_filenames(product, yyyymmdd)
    if len(filenames) == 0:
        logger.warning('Cannot find files of %s on %s', product, yyyymmdd)
        return pd.DataFrame()
    px_list = [pd.read_csv(x, encoding=ENCODING) for x in filenames]
    total_qty = [x.iloc[-1]['总量'] for x in px_list]
    px = px_list[total_qty.index(max(total_qty))]  # select the contract with max qty
    px = process_raw_table(px)
    return px


def load_active_contract_multiple_dates(product, dates):
    logger.info('Loading %s active contract data from %s to %s', product, dates[0], dates[-1])
    px_list = [load_active_contract(product, x) for x in dates]
    px = pd.concat(px_list)
    return px
