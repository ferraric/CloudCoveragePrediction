import os, sys
import tensorflow as tf

from base.base_train import BaseTrain
from utils.dirs import list_files_in_directory
from losses.crps_norm_loss import CrpsNormLoss
from losses.crps_ensemble_loss import CrpsEnsembleLoss


class CRPSTrainer(BaseTrain):
    def __init__(self, model, data, config, comet_logger):
        super(CRPSTrainer, self).__init__(
            model, data, config, comet_logger
        )
        self.optimizer = tf.keras.optimizers.get(self.config.optimizer)
        self.comet_logger = comet_logger
        self.setup_metrics()

    def train(self):
        super().train()
        self.evaluate_ensemble_crps()

    def setup_metrics(self):
        self.loss_object = CrpsNormLoss()
        self.ensemble_loss = CrpsEnsembleLoss()
        self.best_loss = sys.maxsize

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")

        self.validation_loss = tf.keras.metrics.Mean(name="validation_loss")
        self.validation_ensemble_loss =  tf.keras.metrics.Mean(name="validation_ensemble_loss")

    def train_epoch(self):
        for step, (x_batch, y_batch) in enumerate(self.data.train_data):
            if (step % self.config.validate_every_x_batches == 0) and (step != 0):
                self.validation_step()

            self.train_step(x_batch, y_batch)

        self.train_loss.reset_states()

    def train_step(self, x_batch, y_batch):
        with self.comet_logger.train():
            with tf.GradientTape() as tape:
                predictions = self.model(x_batch)
                loss = self.loss_object(y_batch, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            self.train_loss(loss)
            self.comet_logger.log_metric("loss", loss, step=self.optimizer.iterations)
            self.comet_logger.log_metric(
                "average_loss", self.train_loss.result(), step=self.optimizer.iterations
            )

    def validation_step(self):
        self.validation_loss.reset_states()
        self.validation_ensemble_loss.reset_states()

        with self.comet_logger.test():
            for (x_batch, y_batch) in self.data.validation_data:
                predictions = self.model(x_batch)
                loss = self.loss_object(y_batch, predictions)

                self.validation_loss(loss)

            self.comet_logger.log_metric(
                "average_loss", self.validation_loss.result(), step=self.optimizer.iterations
            )


            if self.validation_loss.result() < self.best_loss:
                self.best_loss = self.validation_loss.result()
                model_files = list_files_in_directory(self.config.checkpoint_dir)
                self.save_model()
                for f in model_files:
                    os.remove(f)

    def evaluate_ensemble_crps(self):
        with self.comet_logger.test():
            for (x_batch, y_batch) in self.data.validation_data:
                predictions = self.model(x_batch)
                ensemble_loss = self.ensemble_loss(y_batch, predictions)
                self.validation_ensemble_loss(ensemble_loss)

            self.comet_logger.log_metric(
                "ensemble_loss", self.validation_ensemble_loss.result(), step=self.optimizer.iterations
            )

    def save_model(self):
        tf.saved_model.save(
            self.model,
            os.path.join(
                self.config.checkpoint_dir,
                "model_at_iter_" + str(self.optimizer.iterations),
            ),
        )
