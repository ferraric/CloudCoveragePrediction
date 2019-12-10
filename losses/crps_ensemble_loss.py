import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.losses import Loss
from properscoring import crps_ensemble
from scipy.stats import norm

class CrpsEnsembleLoss(Loss):
    #EPS = 0.0001
    # TODO: type annotations
    def call(self, y_true, a):
        """Compute the CRPS cost function for 21 ensemble members sampled from a normal distribution,
        given by the mean and variance. The distribution is truncated since cloud coverage predicitions are in
        the range [0, 100].

            Args:
                y_true: True values
                y_pred: Tensor containing predictions: [mean, var]
            Returns:
                mean_crps: Scalar with mean CRPS over batch
            """
        # TODO: refactor this
        y_true = tf.reshape(y_true,[tf.shape(y_true)[0]])
        #y_pred = tf.reshape(y_pred, [-1, 2])
        #print(tf.shape(y_pred))

        mu = tf.reshape(a[0],[tf.shape(a[0])[0]])
        mu = tf.clip_by_value(mu, clip_value_min=0, clip_value_max=100)
        var = tf.reshape(a[1],[tf.shape(a[1])[0]])
        # since model might predict negative var
        stdev = tf.math.sqrt(tf.math.abs(var))
        #var = tf.clip_by_value(var, clip_value_min=self.EPS, clip_value_max=np.inf)
        stdev = tf.clip_by_value(stdev, clip_value_min=0.0001, clip_value_max=np.inf)
        no_quantiles = 21
        quantiles = [(i - 0.5) / no_quantiles for i in range(1, no_quantiles + 1)]
        quantiles = np.broadcast_to(quantiles, (mu.shape[0], no_quantiles))

        y_ensemble_pred = np.transpose(norm.ppf(np.transpose(quantiles), loc=mu, scale=stdev))
        y_ensemble_pred = tf.clip_by_value(y_ensemble_pred, clip_value_min=0, clip_value_max=100)

        #y_ensemble_pred = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=var) \
        #    .quantile(tf.convert_to_tensor([(i - 0.5) / no_quantiles for i in range(1, no_quantiles + 1)]))

        crps = crps_ensemble(y_true, y_ensemble_pred)
        return tf.math.reduce_mean(crps)

