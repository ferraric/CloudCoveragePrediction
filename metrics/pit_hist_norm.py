import numpy as np
from scipy.stats import norm
import json


class PitHistNorm():
    EPS = 0.0001

    def __init__(self, name):
        self.name = name
        self.reset_states()

    def update_hist(self, y_true, y_pred):
        cdf_values = self.compute_cdf_values(y_true, y_pred)
        self.append_cdf_values(cdf_values)

    def compute_cdf_values(self, y_true, y_pred):
        y_pred = np.reshape(y_pred, [-1, 2])
        y_true = np.reshape(y_true, (-1, 1))
        assert y_true.shape[0] == y_pred.shape[0]
        y_true = y_true[:, 0]
        mu = y_pred[:, 0]
        mu = np.clip(mu, a_min=0, a_max=100)
        var = y_pred[:, 1]
        stdev = np.sqrt(np.abs(var))
        stdev = np.clip(stdev, a_min=self.EPS, a_max=None)
        cdf_values = norm.cdf(y_true, loc=mu, scale=stdev)
        zero_values_ind = (y_true <= 0)
        cdf_values[zero_values_ind] = np.random.uniform(
            low=0,
            high=cdf_values[zero_values_ind],
            size=cdf_values[zero_values_ind].shape)
        hundred_values_ind = (y_true >= 100)
        cdf_values[hundred_values_ind] = np.random.uniform(
            low=cdf_values[hundred_values_ind],
            high=1,
            size=cdf_values[hundred_values_ind].shape)
        return cdf_values

    def result(self):
        return self.cdf_values

    def result_as_json(self):
        hist = np.histogram(self.cdf_values, bins=990, range=(0, 1))
        hist_values = list(hist[0].astype(np.float64))
        return json.dumps(hist_values)

    def append_cdf_values(self, cdf_values):
        if self.cdf_values is None:
            self.cdf_values = cdf_values
        else:
            self.cdf_values = np.append(self.cdf_values, cdf_values)

    def reset_states(self):
        self.cdf_values = None
