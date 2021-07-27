import logging

import numpy as np
import pandas as pd


def get_date_feats(data: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    new_data = data.copy()
    
    new_data['day_month'] = series.dt.day
    new_data['day_week'] = series.dt.dayofweek
    new_data['day_year'] = series.dt.dayofyear
    new_data['weekofyear'] = series.dt.isocalendar().week.astype(int)
    new_data['quarter'] = series.dt.quarter
    new_data['month'] = series.dt.month
    new_data['year'] = series.dt.year
    
    new_data['sin_day_year'] = np.sin(series.dt.dayofyear)
    new_data['sin_week_year'] = np.sin(series.dt.isocalendar().week.astype(int))
    new_data['sin_month'] = np.sin(series.dt.month)   
    new_data['sin_year'] = np.sin(series.dt.year)
    
    return new_data

def stats_per_period(tr_data: pd.DataFrame,
                     te_data: pd.DataFrame,
                     groupby: list,
                     targets: list) -> tuple:

    prefix = groupby[-1] + '_'
    new_tr = tr_data.copy()
    new_te = te_data.copy()

    stds = tr_data.groupby(groupby).std().reset_index()[groupby + targets]
    stds.columns = groupby + [prefix + col + '_std' for col in stds.columns if col not in groupby]

    means = tr_data.groupby(groupby).mean().reset_index()[groupby + targets]
    means.columns = groupby + [prefix + col + '_mean' for col in means.columns if col not in groupby]

    medians = tr_data.groupby(groupby).median().reset_index()[groupby + targets]
    medians.columns = groupby + [prefix + col + '_median' for col in medians.columns if col not in groupby]

    maxs = tr_data.groupby(groupby).max().reset_index()[groupby + targets]
    maxs.columns = groupby + [prefix + col + '_max' for col in maxs.columns if col not in groupby]

    mins = tr_data.groupby(groupby).min().reset_index()[groupby + targets]
    mins.columns = groupby + [prefix + col + '_min' for col in mins.columns if col not in groupby]

    var_s = tr_data.groupby(groupby).var().reset_index()[groupby + targets]
    var_s.columns = groupby + [prefix + col + '_var' for col in var_s.columns if col not in groupby]

    new_tr = tr_data.merge(var_s, how='left', on=groupby)
    new_te = te_data.merge(var_s, how='left', on=groupby)

    new_tr = new_tr.merge(mins, how='left', on=groupby)
    new_te = new_te.merge(mins, how='left', on=groupby)

    new_tr = new_tr.merge(maxs, how='left', on=groupby)
    new_te = new_te.merge(maxs, how='left', on=groupby)

    new_tr = new_tr.merge(means, how='left', on=groupby)
    new_te = new_te.merge(means, how='left', on=groupby)

    new_tr = new_tr.merge(stds, how='left', on=groupby)
    new_te = new_te.merge(stds, how='left', on=groupby)

    return new_tr, new_te

def clean(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df[df.columns[4:]] = df[df.columns[4:]]\
                                              .apply(lambda x : x.str.replace('\xa0', '').str.replace(',', '.'))\
                                              .astype('float32')
    return new_df

def join(dfs: list) -> pd.DataFrame:
    tmp_df = pd.concat(dfs, axis=1)
    return tmp_df.loc[:, ~tmp_df.columns.duplicated()]

def get_logger(folder: str) -> logging.Logger:
    ff_name = './{}/learn.log'.format(folder)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler(ff_name)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s] - %(asctime)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger