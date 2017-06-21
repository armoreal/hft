import logging

import hft.data_loader as dl
import hft.signal_utils as signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(name)s  %(levelname)s  %(message)s')

product = 'cu'  # switch between cu and zn
seconds = 10
return_col = 'return'+str(seconds)
return_cutoff = 6

# test on a single date
px = dl.load_active_contract(product, '20131231')
px = signal.volume_order_imbalance(px)
px = signal.order_imbalance_ratio(px)
px = signal.period_return(px, 5)
