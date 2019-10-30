from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D
import os


class ExampleModel(Model):
    def __init__(self, config):
        super(ExampleModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.conv1 = Conv2D(2, [2,2], padding="same", activation="relu")

    def call(self, x):
        x = self.conv1(x)
        return x

    def log_model_architecture_to(self, experiment, input_shape):
        self.build(input_shape)
        model_architecture_path = os.path.join(self.config.summary_dir, "model_architecture")
        with open(model_architecture_path, "w") as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.summary(print_fn=lambda x: fh.write(x + "\n"))
        print(self.summary())
        experiment.log_asset(model_architecture_path)