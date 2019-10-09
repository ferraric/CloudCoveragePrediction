import numpy as np
from scipy.stats import norm

import properscoring as ps


def crps_gaussian(y_true, y_pred):
    """
    Calculates the crps score for a model that predicts both mean and
    variance of a gaussian posterior distribution.
    """
    mu = y_pred[0]
    sig = y_pred[1]
    scores = ps.crps_gaussian(y_true, mu=mu, sig=sig)
    return scores.mean()
