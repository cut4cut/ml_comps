import sys
import joblib
import logging

import pandas as pd
import numpy as np
import lightgbm as lgb

from tqdm import tqdm

from lib import get_logger

URL = 'https://assets.codeforces.com/rounds/338595/c510ad677402d8bd/{}.csv'
NAMES = ['train', 'user-features', 'test', 'sample-submission']
N_MACRO_FOLDS = 1000
N_FOLDS = 25

def cross_validate(tr_targs: pd.DataFrame,
                   tr_feats: pd.DataFrame) -> tuple:

    return (0, )

def fit_predict(test: pd.DataFrame,
                logger: logging.Logger) -> pd.DataFrame:
    
    return test

def main():

    folder = sys.argv[0]

    logger = get_logger(folder)

    try:
        mode = sys.argv[1]
    except IndexError:
        logger.warning('Mode is None. Choosed mode=load')
        mode = 'load'
    
    urls = [URL.format(file) for file in NAMES]
    files = ['./{}/data/{}.csv'.format(folder, file) for file in NAMES]

    if mode == 'load':
        logger.info('Start load datasets')
        for i in tqdm(range(len(NAMES))):
            data = pd.read_csv(urls[i])
            data.to_csv(files[i], index=False) 
    else:
        pass


if __name__ == '__main__':

    main()