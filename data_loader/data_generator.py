import numpy as np
import tensorflow as tf


class DataGenerator:
    def __init__(self, config, comet_logger):
        self.config = config
        self.comet_logger = comet_logger

        assert (
            "batch_size" in self.config
        ), "You need to define the parameter 'batch_size' in your config file."
        assert (
            "shuffle_buffer_size" in self.config
        ), "You need to define the parameter 'shuffle_buffer_size' in your config file."

        # dummy input
        inputs = np.zeros((1, 50, 50, 1))
        labels = np.ones((1, 50, 50, 1))

        self.train_data = tf.data.Dataset.from_tensor_slices((inputs, labels))
        self.train_data = self.train_data.shuffle(
            buffer_size=self.config.shuffle_buffer_size).repeat(100)
        self.train_data = self.train_data.batch(self.config.batch_size,
                                                drop_remainder=True)

        # dummy input
        validation_inputs = np.zeros((1, 50, 50, 1))
        validation_labels = np.ones((1, 50, 50, 1))

        self.validation_data = tf.data.Dataset.from_tensor_slices(
            (validation_inputs, validation_labels))
        self.validation_data = self.validation_data.batch(
            self.config.batch_size)

        self.comet_logger.log_dataset_hash(self.train_data)
        self.comet_logger.log_dataset_hash(self.validation_data)
