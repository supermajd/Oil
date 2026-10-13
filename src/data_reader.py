
""" Reads Trump's Truth Social posts (data.json) and the WTI prices (data/oil_indices).
"""

__author__ = 'Majd Jamal'

import pandas as pd

TRUTHS = 'data/posts.json'                     # the posts without their text: id, link, time and author


WTI_RESOLUTION = {'1s': 'data/oil_indices/wti_1s.csv',
                  'tick': 'data/oil_indices/wti_ticks.csv',
                  'posts': 'data/oil_indices/wti_post_prices.csv'}   # the bid 0-5 s and 30-35 s after each oil post (published)

def read_trumps_posts():
    """ Loads data.json and returns only the posts written by Trump himself.
    """
    df = pd.read_json(TRUTHS)

    X = ['id', 'archive_id', 'author_name', 'published_at_utc', 'published_at_et', 'kind']

    df = df[df['is_written_by_trump'] == True][X]
    df['published_at_utc'] = pd.to_datetime(df['published_at_utc'])

    return df

def read_wti(res = 'tick'):
    """
    :param res: Input resolution for the data
    """

    wti = WTI_RESOLUTION[res]
    wti = pd.read_csv(wti)

    wti['time_utc'] = pd.to_datetime(wti.time_utc)
    wti = wti.set_index(wti.time_utc)
    wti = wti.drop(columns = ['time_utc'])

    return wti


if __name__ == "__main__":

    oil_prices = read_wti(res = '1d')
