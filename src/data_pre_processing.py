
""" Turns the GPT labels into model features (pre_process), the prices into
3-class labels (make_labels), and matches each post to its price second (intersect).
"""

__author__ = 'Majd Jamal'

import pandas as pd
import numpy as np

THRESHOLD = 0.075 


UP = ['supply_or_transit_loss', 'sanctions_tighten', 'escalation', 'demand_stronger']
DOWN = ['supply_or_transit_gain', 'sanctions_ease', 'deescalation', 'demand_weaker', 'producer_price_pressure']
NEW = ['proposed_or_threatened', 'ordered_or_implementation_reported']
CANCELLED = 'withdrawn_or_suspended'


def is_up(post):

    if post['text_form'] == 'passed_on' or post['oil_relevance'] == 'none':
        return 0

    if post['action_status'] in NEW and post['oil_mechanism'] in UP:          # new step that pushes up
        return 1

    if post['action_status'] == CANCELLED and post['oil_mechanism'] in DOWN:  # cancelled step that pushes down
        return 1

    return 0


def is_down(post):

    if post['text_form'] == 'passed_on' or post['oil_relevance'] == 'none':
        return 0

    if post['action_status'] in NEW and post['oil_mechanism'] in DOWN:        # new step that pushes down
        return 1

    if post['action_status'] == CANCELLED and post['oil_mechanism'] in UP:    # cancelled step that pushes up
        return 1

    return 0

def pre_process(posts):

    X = pd.DataFrame(index=posts.index)

    X['is_up'] = posts.apply(is_up, axis=1)
    X['is_down'] = posts.apply(is_down, axis=1)

    X['iran_sign'] = np.where(posts['counterpart'] == 'iran', 1, -1)
    X['other_sign'] = np.where(posts['counterpart'] != 'iran', 1, -1)
    
    X['is_direct'] = np.where(posts['oil_relevance'] == 'direct', 1, -1)
    X['is_us_direct'] = posts['us_acts']
    
    X['mech_escalation'] = np.where(posts['oil_mechanism'] == 'escalation', 1, 0)
    X['mech_deescalation'] = np.where(posts['oil_mechanism'] == 'deescalation', 1, 0)
    X['mech_supply_or_transit_loss'] = np.where(posts['oil_mechanism'] == 'supply_or_transit_loss', 1, 0)
    X['mech_supply_or_transit_gain'] = np.where(posts['oil_mechanism'] == 'supply_or_transit_gain', 1, 0)
    X['mech_sanctions_tighten'] = np.where(posts['oil_mechanism'] == 'sanctions_tighten', 1, 0)
    X['mech_sanctions_ease'] = np.where(posts['oil_mechanism'] == 'sanctions_ease', 1, 0)
    X['mech_demand_weaker'] = np.where(posts['oil_mechanism'] == 'demand_weaker', 1, 0)
    X['mech_demand_stronger'] = np.where(posts['oil_mechanism'] == 'demand_stronger', 1, 0)

    X['status_proposed_or_threatened'] = np.where(posts['action_status'] == 'proposed_or_threatened', 1, 0)
    X['status_ordered_or_implementation_reported'] = np.where(posts['action_status'] == 'ordered_or_implementation_reported', 1, 0)
    X['status_withdrawn_or_suspended'] = np.where(posts['action_status'] == 'withdrawn_or_suspended', 1, 0)
    
    return X 




def make_labels(prices, posts, delay = '0s', window = '30s'):

    post_time = pd.to_datetime(posts['id'].astype('int64').to_numpy() >> 16, unit='ms', utc=True)
    

    start = post_time + pd.Timedelta(delay)
    
    bid = prices['bid']


    entry = bid.reindex(start, method='ffill', tolerance=pd.Timedelta('60s')).values 

    exit = bid.reindex(start + pd.Timedelta(window), method='ffill', tolerance=pd.Timedelta('60s')).values

    y = pd.DataFrame(index=posts.index)

    y['percent'] = (exit / entry - 1) * 100

    y['sign'] = np.where(y['percent'].abs() <= THRESHOLD, 0, np.sign(y['percent']))

    return y


def intersect(X, y):

    X = X.copy()
    X.index = X.index.floor('1s')        
    X = X[X.index.isin(y.index)]

    return X, y.loc[X.index]