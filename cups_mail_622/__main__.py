import os
import re
import sys 
import csv
import lib
import shutil

from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile

EXTRACT = True

def cross_validate():
    pass

def fit_predict():
    pass

def main():
    folder = sys.argv[0]

    logger = lib.get_logger(folder)

    zips = ['./{}/data/avia-{}.zip'.format(folder, file) for file in ['train', 'test']]
    files = ['./{}/data/{}.csv'.format(folder, file) for file in ['train', 'submission']]
    folders = ['./{}/data/{}'.format(folder, file) for file in ['train', 'test']]
    labels = {1: 'airplane', 0: 'not_airplane'}

    for f in folders:
        Path(f).mkdir(parents=True, exist_ok=True)
        if 'test' in f: break
        for l in list(labels.values()):
            Path(f + '/' + l).mkdir(parents=True, exist_ok=True) 
            
    if EXTRACT:
        logger.info('Start extract train data')
        lib.extract_train(folders[0], files[0], zips[0], labels)
        logger.info('Start extract test data')
        lib.extract_test(folders[1], zips[1])



if __name__ == '__main__':

    main()