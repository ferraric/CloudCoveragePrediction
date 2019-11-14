import tensorflow as tf
from tensorflow.keras import Model
import os


class IdentityModel(Model):
    def __init__(self, config):
        super(IdentityModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        pass

    def call(self, x):
        return x

    def log_model_architecture_to(self, experiment, input_shape):
        self.build(input_shape)
        model_architecture_path = os.path.join(self.config.summary_dir, "model_architecture")
        with open(model_architecture_path, "w") as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.summary(print_fn=lambda x: fh.write(x + "\n"))
        self.summary()
        experiment.log_asset(model_architecture_path)