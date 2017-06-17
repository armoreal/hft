import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_PATH = 'C:\\Users\\Jingwei\\Documents\\HFFut\\SpRawFutureTick'
COLUMNS = ['date', 'time', 'price', 'qty', 'total', 'outstanding_change',
           'b1', 'b1_size', 'b2', 'b2_size', 'b3', 'b3_size',
           's1', 's1_size', 's2', 's2_size', 's3', 's3_size', 'bs']


def get_dates():
    return os.listdir(DATA_PATH)


def get_filenames(product, yyyymmdd):
    contract_month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    filenames = [os.path.join(DATA_PATH, yyyymmdd, product + x + '_' + yyyymmdd + '.csv') for x in contract_month]
    filenames = [x for x in filenames if os.path.isfile(x)]
    return filenames


def load_contract(product, yyyymmdd, contract_month):
    filename = os.path.join(DATA_PATH, yyyymmdd, product + contract_month + '_' + yyyymmdd + '.csv')
    px = pd.read_csv(filename, encoding='gb18030')
    px.columns = COLUMNS
    px['dt'] = px['date'] + ' ' + px['time']
    px['dt'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in px['dt']]
    px.set_index('dt', inplace=True)
    return px


def load_active_contract(product, yyyymmdd):
    filenames = get_filenames(product, yyyymmdd)
    if len(filenames) == 0:
        logger.warning('Cannot find files of %s on %s', product, yyyymmdd)
        return pd.DataFrame()
    px_list = [pd.read_csv(x, encoding="gb18030") for x in filenames]
    total_qty = [x.iloc[-1]['总量'] for x in px_list]
    px = px_list[np.argmax(total_qty)]  # select the contract with max qty
    px.columns = COLUMNS
    px['dt'] = px['date'] + ' ' + px['time']
    px['dt'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in px['dt']]
    px.set_index('dt', inplace=True)
    return px


def load_active_contract_multiple_dates(product, dates):
    px_list = [load_active_contract(product, x) for x in dates]
    px = pd.concat(px_list)
    return px
