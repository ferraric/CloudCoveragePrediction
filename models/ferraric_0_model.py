import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
import os


class CNNModel(Model):
    def __init__(self, config):
        super(CNNModel, self).__init__()
        self.config = config
        self.build_model()

    def build_model(self):
        self.pad = ZeroPadding2D(padding=((1, 0), (0, 0)))
        self.conv1 = Conv2D(121, [3,3], padding="same", activation="relu", input_shape=(90, 59, 121))
        self.max_pool = MaxPooling2D(padding="same")
        self.conv2 = Conv2D(121, [3,3], padding="same", activation="relu")
        self.up_sam = UpSampling2D((2, 2))
        self.crop = Cropping2D(cropping=((1, 1), (1, 0)))

    def call(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.up_sam(x)
        x = self.crop(x)
        return x

    def log_model_architecture_to(self, experiment, input_shape):
        self.build(input_shape)
        model_architecture_path = os.path.join(self.config.summary_dir, "model_architecture")
        with open(model_architecture_path, "w") as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.summary(print_fn=lambda x: fh.write(x + "\n"))
        self.summary()
        experiment.log_asset(model_architecture_path)