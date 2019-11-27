import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from properscoring import crps_ensemble
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

class Crps21EnsembleLoss(Loss):
    # TODO: type annotations
    def call(self, y_true, y_pred):
        """Compute the CRPS cost function for 21 ensemble members.

            Args:
                y_true: True values
                y_pred: Tensor containing 21 ensemble predictions
            Returns:
                mean_crps: Scalar with mean CRPS over batch
            """
        # TODO: refactor this
        y_true = tf.reshape(y_true, [-1,21])
        y_true = y_true[:, 0]  # Need to also get rid of axis 1 to match!
        y_pred = tf.reshape(y_pred, [-1, 21])

        n = y_pred.shape[0]
        d = y_pred.shape[1]
        y_true = tf.reshape(y_true, [n, 1])
        diffs = tf.math.reduce_sum(tf.math.abs(tf.math.subtract(y_pred, tf.broadcast_to(y_true, [n, d]))), axis=1)
        strech_terms = np.apply_along_axis(
            lambda row: tf.math.reduce_sum(squareform(pdist(tf.reshape(row, [d, 1]), 'cityblock'))),
            axis=1, arr=y_pred)
        new_crps =  diffs / d - strech_terms / (2 * d ** 2)
        #crps = crps_ensemble(np.transpose(y_true), np.transpose(y_pred))
        #assert new_crps == crps
        return tf.math.reduce_mean(new_crps)
