import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LocallyConnected2D
import os


class SimpleModel(Model):
    def __init__(self, config):
        super(SimpleModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.w1 = tf.Variable(tf.random.normal(shape=[90, 59, 121], mean=1.0, stddev=1.0));
        self.b1 = tf.Variable(tf.random.normal(shape=[90, 59, 121]));

    def call(self, x):
        return tf.math.add(tf.math.multiply(self.w1, x), self.b1)

class LocalModel(Model):
    def __init__(self, config):
        super(LocalModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.loc = LocallyConnected2D(filters=121, kernel_size=[1,1], use_bias=True, input_shape=(90, 59, 121))

    def call(self, x):
        x = self.loc(x)
        return x

    def log_model_architecture_to(self, experiment, input_shape):
        self.build(input_shape)
        model_architecture_path = os.path.join(self.config.summary_dir, "model_architecture")
        with open(model_architecture_path, "w") as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.summary(print_fn=lambda x: fh.write(x + "\n"))
        self.summary()
        experiment.log_asset(model_architecture_path)