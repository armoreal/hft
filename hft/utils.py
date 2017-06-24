"""
Utility functions
"""

# table aggregation
# -----------------


def aggregate(pxall, group, funs, rename_dict=None):
    daily_agg_px = pxall.groupby(['date', group]).agg(funs)
    daily_agg_px.rename(columns=rename_dict, inplace=True)
    daily_agg_px['n_trades'] = pxall.groupby(['date', group]).size()
    agg_px = daily_agg_px.reset_index().groupby(group).median()
    return agg_px


# compute a new column based on a period of data
# ----------------------------------------------


def get_moving_column_name(column, backward_seconds, forward_seconds):
    return column + '_' + str(backward_seconds) + '_' + str(forward_seconds)


def get_index_within_period(px, backward_seconds, forward_seconds):
    px['forward_second'] = px.second + forward_seconds
    px['backward_second'] = px.second - backward_seconds
    n = px.shape[0]
    idx_col = get_moving_column_name('index_within_period', backward_seconds, forward_seconds)
    px[idx_col] = [((px.second <= px.forward_second[i]) & (px.second >= px.backward_second[i])).values
                   for i in range(n)]
    px.drop(['forward_second', 'backward_second'], axis=1, inplace=True)
    return px


def moving_operate(px, column, fun, backward_seconds, forward_seconds, new_column=None):
    """Compute the moving operation of a column

    :param px: pandas data frame, need to have column column
    :type px: pandas data frame
    :param forward_seconds: int, number of seconds going forward
    :param backward_seconds: int, number of seconds going backward
    :param column: string, column name
    :param fun: function, could be average, sum or any user-defined operations
    :param new_column: string, new column name
    :return: pandas data frame
    """
    idx_col = get_moving_column_name('index_within_period', backward_seconds, forward_seconds)
    if idx_col not in px.columns:
        px = get_index_within_period(px, backward_seconds, forward_seconds)
    if new_column is None:
        new_column = column
    new_column = get_moving_column_name(new_column, backward_seconds, forward_seconds)
    px[new_column] = [fun(px[column][idx]) for idx in px[idx_col]]
    return px
