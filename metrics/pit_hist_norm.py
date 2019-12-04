import numpy as np
from scipy.stats import norm

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
        var = np.clip(var, a_min=self.EPS, a_max=None)
        cdf_values = norm.cdf(y_true, loc=mu, scale=var)
        zero_values_ind = (y_true <= 0)
        cdf_values[zero_values_ind] = np.random.uniform(low=0, high=cdf_values[zero_values_ind],
                                                        size=cdf_values[zero_values_ind].shape)
        hundred_values_ind = (y_true >= 100)
        cdf_values[hundred_values_ind] = np.random.uniform(low=cdf_values[hundred_values_ind], high=1,
                                                           size=cdf_values[hundred_values_ind].shape)
        np.unique(np.argwhere(cdf_values < 0))
        return cdf_values

    def result(self):
        return self.cdf_values

    def append_cdf_values(self, cdf_values):
        if self.cdf_values is None:
            self.cdf_values = cdf_values
        else:
            self.cdf_values = np.append(self.cdf_values, cdf_values)

    def reset_states(self):
        self.cdf_values = None