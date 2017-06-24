import logging

import hft.data_loader as dl
import hft.signal_utils as signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

product = 'cu'  # switch between cu and zn
tick_size = 10 if product == 'cu' else 5
seconds = 10
return_col = 'return'+str(seconds)
return_cutoff = 6

# test on a single date
px = dl.load_active_contract(product, '20131231')
px = signal.order_flow_imbalance(px, 1, 0)
px = signal.order_imbalance_ratio(px, 1, 0)
px = signal.period_mid_move(px, 5, 0, tick_size)
px = signal.period_mid_move(px, 0, 10, tick_size)
