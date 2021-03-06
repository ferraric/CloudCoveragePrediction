import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss


class CrpsNormLoss(Loss):
    EPS = 0.0001

    # TODO: type annotations
    def call(self, y_true, a):
        """Compute the CRPS cost function for a normal distribution defined by
            the mean and variance.
            Code taken and adapted from https://github.com/slerch/ppnn.
            Args:
                y_true: True values
                y_pred: Tensor containing predictions: [mean, var]
            Returns:
                mean_crps: Scalar with mean CRPS over batch
        """
        # TODO: refactor this
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0]])
        #y_pred = tf.reshape(y_pred, [-1, 2])
        #print(tf.shape(y_pred))
        #print(a.shape)
        mu = tf.reshape(a[0], [tf.shape(a[0])[0]])
        #mu = tf.clip_by_value(mu, clip_value_min=0, clip_value_max=100)
        var = tf.reshape(a[1], [tf.shape(a[1])[0]])
        print(tf.shape(y_true))
        print(tf.shape(mu))
        print(tf.shape(var))
        #y_true = y_true[:, 0]  # Need to also get rid of axis 1 to match!

        # since model might predict negative var
        var = tf.math.abs(var)
        var = tf.clip_by_value(var,
                               clip_value_min=0.0001,
                               clip_value_max=np.inf)

        # The following three variables are just for convenience
        loc = (y_true - mu) / tf.math.sqrt(var)
        phi = 1.0 / np.sqrt(2.0 * np.pi) * tf.math.exp(
            -tf.math.square(loc) / 2.0)
        Phi = 0.5 * (1.0 + tf.math.erf(loc / np.sqrt(2.0)))
        # First we will compute the crps for each input/target pair
        crps = tf.math.sqrt(var) * (loc * (2. * Phi - 1.) + 2 * phi -
                                    1. / np.sqrt(np.pi))
        # Then we take the mean. The cost is now a scalar
        return tf.math.reduce_mean(crps)
