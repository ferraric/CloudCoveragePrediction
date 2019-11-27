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
        y_true = tf.reshape(y_true, [-1,21])
        y_true = y_true[:, 0]
        y_pred = tf.reshape(y_pred, [-1, 21])

        n = tf.shape(y_pred)[0]
        d = tf.shape(y_pred)[1]
        y_true = tf.reshape(y_true, [n, 1])
        diffs = tf.math.reduce_sum(tf.math.abs(tf.math.subtract(y_pred, tf.broadcast_to(y_true, (n, d)))), axis=1)
        pair_diff_sum = tf.map_fn(lambda row: tf.reduce_sum(tf.abs(tf.subtract(row, tf.expand_dims(row, 1)))), y_pred)
        crps =  diffs / d - pair_diff_sum / (2 * d ** 2)

        return tf.math.reduce_mean(crps)
