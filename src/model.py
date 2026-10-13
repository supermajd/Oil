
""" The classifier (a small gradient-boosted tree model with fixed settings)
and the walk-forward settings.
"""

__author__ = 'Majd Jamal'

from sklearn.ensemble import HistGradientBoostingClassifier


MIN_TRAIN = 30                      # walk forward: posts needed before the first test month
FIRST_TEST_MONTH = '2026-02'        # walk forward: months from here are tested; earlier posts only train


PARAMS = {
    'learning_rate': 0.03,          # small steps: each tree corrects the previous ones only a little
    'max_depth': 2,                 # shallow trees: at most two questions per tree
    'max_leaf_nodes': 8,
    'min_samples_leaf': 10,         # every answer must be backed by at least 10 posts
    'l2_regularization': 1.0,       # shrinks extreme leaf values
    'max_iter': 100,                # number of trees
    'class_weight': 'balanced',     # up, flat and down count equally, even if one is more common
    'random_state': 0,
}

def build_model():
    """ Returns an unfitted classifier for the per-post target: the WTI move in the 30 seconds after the post, up (1), flat (0) or down (-1).
    The settings are fixed in advance, not tuned.
    """
    return HistGradientBoostingClassifier(**PARAMS)


def train(model, X_train, y_train):
    model = model.fit(X_train, y_train)
    return model
