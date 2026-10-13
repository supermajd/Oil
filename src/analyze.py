
""" Runs the study: labels each oil post with the WTI move 30 s after it (up, no move, down),
tests the model month by month (walk forward) and prints the macro F1.

Run from the project root:  python3 src/analyze.py
"""

__author__ = 'Majd Jamal'


import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

from data_reader import read_trumps_posts, read_wti
from data_pre_processing import pre_process, make_labels, intersect
from model import build_model, MIN_TRAIN, FIRST_TEST_MONTH

FEATURES = 'data/features.json'

resolution = 'posts'

def read_features():

    if not os.path.exists(FEATURES):
        raise FileNotFoundError('Features not computed yet! Run a batch with: python3 src/feature_engineering.py')

    ft = pd.read_json(FEATURES)
    truths = read_trumps_posts()

    ft = ft.merge(truths, on='id')
    ft = ft.set_index('published_at_utc')

    return ft

def read_price():

    return read_wti(resolution)


def walk_forward(X, Y):
    
    
    preds = pd.Series(np.nan, index=X.index)

    months = X.index.strftime('%Y-%m')  
    
    for month in tqdm(sorted(set(months))):

        hgb = build_model()

        X_train = X[months < month]
        X_test = X[months == month]

        Y_train = Y[months < month]

        if month < FIRST_TEST_MONTH or len(Y_train) < MIN_TRAIN or Y_train.nunique() < 2:
                    continue

        hgb.fit(X_train, Y_train)

        preds[months == month]  = hgb.predict(X_test)

    return preds

if __name__ == '__main__':

    features = read_features()
    features = features[features['oil_relevance'] != 'none']     # only posts about oil
    X = pre_process(features)


    prices = read_price()
    Y = make_labels(prices, features, delay = '0s', window='30s')['sign']

    keep = Y.notna().values          # only posts with a price before and after
    X, Y = X[keep], Y[keep]
    
    print(X.shape)
    preds = walk_forward(X, Y)

    tested = preds.notna()
    print(tested.shape)
    print(f1_score(Y[tested], preds[tested], labels=[-1, 1], average='macro'))
