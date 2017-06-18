"""
Utility functions
"""

# table manipulation
# ------------------


def aggregate(pxall, group, funs, rename_dict=None):
    daily_agg_px = pxall.groupby(['date', group]).agg(funs)
    daily_agg_px.rename(columns=rename_dict, inplace=True)
    daily_agg_px['count'] = pxall.groupby(['date', group]).size()
    agg_px = daily_agg_px.reset_index().groupby(group).median()
    return agg_px
