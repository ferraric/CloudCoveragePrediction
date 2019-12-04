import numpy as np

class PitHist():
    def __init__(self, name):
        self.name = name
        self.reset_states()

    def update_hist(self, y_true, y_pred):
        ranks = self.compute_ranks(y_true, y_pred)
        self.append_ranks(ranks)

    def compute_ranks(self, y_true, y_pred):
        y_pred = np.reshape(y_pred, [-1, 2])
        y_pred = np.clip(y_pred, a_min=0, a_max=100)
        y_true = np.reshape(y_true, (-1, 1))
        assert y_true.shape[0] == y_pred.shape[0]
        all_values = np.append(y_pred, y_true, axis=-1)
        corner_cases_ind = ((all_values == 0.0) | (all_values == 100.0))
        all_values[corner_cases_ind] = all_values[corner_cases_ind] + 0.001 * np.random.standard_normal(
            all_values[corner_cases_ind].shape)
        ranks = all_values.argsort(axis=-1).argsort(axis=-1)
        return ranks[:, -1]

    def result(self):
        return self.ranks

    def append_ranks(self, ranks):
        if self.ranks is None:
            self.ranks = ranks
        else:
            self.ranks = np.append(self.ranks, ranks)

    def reset_states(self):
        self.ranks = None