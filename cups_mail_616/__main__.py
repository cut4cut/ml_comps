import sys
import joblib
import logging

import pandas as pd
import numpy as np
import lightgbm as lgb

from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dim import TARGETS
from lib import get_logger, clean, join, get_date_feats, stats_per_period

N_MACRO_FOLDS = 1000
N_FOLDS = 25

def cross_validate(tr_targs: pd.DataFrame,
                   tr_feats: pd.DataFrame) -> tuple:

    cv_res = {}
    len_targs = len(TARGETS)

    cv_mse = np.zeros(len_targs)
    cv_mae = np.zeros(len_targs)
    cv_l_e = np.zeros(len_targs)

    tscv = TimeSeriesSplit(n_splits=N_FOLDS, max_train_size=180, test_size=90)

    for i, t in tqdm(enumerate(TARGETS)):
    
        mask = tr_targs[t] == 0
        
        tr_Y = tr_targs[t][~mask].reset_index(drop=True)    
        tr_X = tr_feats[:][~mask].reset_index(drop=True)
        
        cf_mse = np.ones(N_FOLDS)
        cf_mae = np.ones(N_FOLDS)
        cf_l_e = np.ones(N_FOLDS)
        
        cv_ = {'pred': [], 'valid': []}
        
        for j, (tr_idx, vl_idx) in enumerate(tscv.split(tr_X, tr_Y)):
        
            tr_X_, tr_Y_ = tr_X.loc[tr_idx], tr_Y.loc[tr_idx]
            vl_X_, vl_Y_ = tr_X.loc[vl_idx], tr_Y.loc[vl_idx]

            model = lgb.LGBMRegressor(num_leaves= 50, n_estimators=200)
            model.fit(tr_X_, tr_Y_)
            pred = model.predict(vl_X_)
            
            mse = mean_squared_error(vl_Y_, pred)
            mae = mean_absolute_error(vl_Y_, pred)
            mean = np.mean(vl_Y_)
            
            cf_mse[j] = mse
            cf_mae[j] = mae
            cf_l_e[j] = mae/mean
        
            cv_['pred'].append(pred)
            cv_['valid'].append(vl_Y_)
            
        cv_res[t] = cv_

        cv_mse[i] = np.mean(cf_mse)
        cv_mae[i] = np.mean(cf_mae)
        cv_l_e[i] = np.mean(cf_l_e)

    return (cv_res, cv_mse, cv_mae, cv_l_e)

def fit_predict(tr_targs: pd.DataFrame, 
                tr_feats: pd.DataFrame,
                te_feats: pd.DataFrame,
                test: pd.DataFrame,
                logger: logging.Logger) -> pd.DataFrame:

    submission = test.copy()

    for t in TARGETS:

        model = lgb.LGBMRegressor(num_leaves=50, n_estimators=200)
        mask = tr_targs[t] == 0

        tr_Y = tr_targs[t][~mask].reset_index(drop=True)    
        tr_X = tr_feats[~mask].reset_index(drop=True)
        te_X = te_feats.reset_index(drop=True)

        model.fit(tr_X, tr_Y)
        pred = model.predict(tr_X)

        mse = mean_squared_error(tr_Y, pred)
        mae = mean_absolute_error(tr_Y, pred)
        smae = mae/np.mean(tr_Y)

        submission[t] = model.predict(te_X)
        
        logger.info('For {:>24} MSE={:>7.3f}, MAE={:>7.3f}, sMAE={:>5.3f}'.format(t, mse, mae, smae))
    
    return submission


def main():

    folder = sys.argv[0]

    logger = get_logger(folder)

    try:
        mode = sys.argv[1]
    except IndexError:
        logger.warning('Mode is None. Choosed mode=cv')
        mode = 'cv'
    
    files = ['./{}/data/{}.csv'.format(folder, file) for file in ['train', 'test', 'submission']]

    logger.info('Start load dataset')
    train = pd.read_csv(files[0], delimiter=';')
    test  = pd.read_csv(files[1])

    logger.info('Cleaning dataset and upd type of cols')
    train = clean(train)
    train['date'] = pd.to_datetime(train['date'],infer_datetime_format=True)
    test['date'] = pd.to_datetime(test['date'],infer_datetime_format=True)

    logger.info('Generating date feats')
    tr_data = get_date_feats(train, train['date'])
    te_data = get_date_feats(test, test['date'])

    logger.info('Generating stat feats')
    tr_feat_regyear, te_feat_regyear = stats_per_period(tr_data, te_data, ['region', 'year'], targets=TARGETS)
    tr_feat_regmon, te_feat_regmon = stats_per_period(tr_data, te_data, ['region', 'month'], targets=TARGETS)
    tr_feat_regquar, te_feat_regquar = stats_per_period(tr_data, te_data, ['region', 'quarter'], targets=TARGETS)
    tr_feat_regdmon, te_feat_regdmon = stats_per_period(tr_data, te_data, ['region', 'month', 'day_month'], targets=TARGETS)

    logger.info('Generating final datasets')
    tr_dataset = join([tr_feat_regdmon, tr_feat_regquar, tr_feat_regmon, tr_feat_regyear])
    te_dataset = join([te_feat_regdmon, te_feat_regquar, te_feat_regmon, te_feat_regyear])

    logger.info('Split datasets on feats and targets')
    tr_feats = tr_dataset.drop(columns=TARGETS + ['date']).copy()
    te_feats = te_dataset.drop(columns=TARGETS + ['date']).copy()
    tr_targs = tr_dataset[TARGETS].copy()

    if mode == 'cv':
        logger.info('Start cross-validation')
        cv_res, cv_mse, cv_mae, cv_l_e = cross_validate(tr_targs, tr_feats)

        mean_mse, std_mse = np.mean(cv_mse), np.std(cv_mse)
        mean_mae, std_mae = np.mean(cv_mae), np.std(cv_mae)
        mean_l_e, std_l_e = np.mean(cv_l_e), np.std(cv_l_e)

        wors_l_e = np.max(cv_l_e)
        best_l_e = np.min(cv_l_e)

        targ_wors = TARGETS[np.argmax(cv_l_e)]
        targ_best = TARGETS[np.argmin(cv_l_e)]

        logger.info('MSE: mean={:2.3f}, std={:2.3f}'.format(mean_mse, std_mse))
        logger.info('MAE: mean={:2.3f}, std={:2.3f}'.format(mean_mae, std_mae))
        logger.info('L_e: mean={:2.3f}, std={:2.3}'.format(mean_l_e, std_l_e))
        logger.info('Final scroe={:2.6f}'.format(1 / (1000*mean_l_e)))
        logger.info('Wors targ={} with L_e={:2.3f}'.format(targ_wors, wors_l_e))
        logger.info('Best targ={} with L_e={:2.3f}'.format(targ_best, best_l_e))

        joblib.dump(cv_res, './report/cv_results.pkl')

    elif mode == 'fit':
        logger.info('Start fit predict')
        submission = fit_predict(tr_targs, tr_feats, te_feats, test, logger)
        submission.to_csv('./data/submission.csv', index=False)
        logger.info('Submission saved')

if __name__ == '__main__':

    main()