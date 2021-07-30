import os, sys
import tensorflow as tf
from base.base_train import BaseTrain
from utils.dirs import list_files_in_directory
from losses.crps_norm_loss import CrpsNormLoss
from losses.crps_ensemble_loss import CrpsEnsembleLoss
from losses.crps_21_ensemble_loss import Crps21EnsembleLoss
from metrics.pit_hist import PitHist
from metrics.pit_hist_norm import PitHistNorm
class CRPSTrainer(BaseTrain):
    def __init__(self, model, data, config, comet_logger):
        super(CRPSTrainer, self).__init__(
            model, data, config, comet_logger
        )
        self.optimizer = tf.keras.optimizers.Adam(0.0001)
        self.comet_logger = comet_logger
        self.setup_metrics()

    def train(self):
        super().train()
    def setup_metrics(self):
        self.loss_object = Crps21EnsembleLoss()
        self.best_loss = sys.maxsize
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validation_pit_hist_win = PitHist(name="validation_pit_hist_win")
        self.validation_pit_hist_spr= PitHist(name="validation_pit_hist_spr")
        self.validation_pit_hist_sum = PitHist(name="validation_pit_hist_sum")
        self.validation_pit_hist_fal = PitHist(name="validation_pit_hist_fal")
        self.validation_loss_win= tf.keras.metrics.Mean(name="validation_loss_winter")
        self.validation_ensemble_loss_win=  tf.keras.metrics.Mean(name="validation_ensemble_loss_winter")
        self.validation_loss_spr = tf.keras.metrics.Mean(name="validation_loss_spring")
        self.validation_ensemble_loss_spr = tf.keras.metrics.Mean(name="validation_ensemble_loss_spring")
        self.validation_loss_sum = tf.keras.metrics.Mean(name="validation_loss_summer")
        self.validation_ensemble_loss_sum = tf.keras.metrics.Mean(name="validation_ensemble_loss_summer")
        self.validation_loss_fal = tf.keras.metrics.Mean(name="validation_loss_fall")
        self.validation_ensemble_loss_fal = tf.keras.metrics.Mean(name="validation_ensemble_loss_fall")

    def train_epoch(self):
        for step, (x_batch, y_batch) in enumerate(self.data.train_data):
            self.train_step(x_batch, y_batch)
            if (step%200==0)&(step!=0):
                self.validation_step()
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
        self.validation_loss_win.reset_states()
        self.validation_pit_hist_win.reset_states()
        self.validation_pit_hist_spr.reset_states()
        self.validation_pit_hist_sum.reset_states()
        self.validation_pit_hist_fal.reset_states()
        self.validation_loss_spr.reset_states()
        self.validation_loss_sum.reset_states()
        self.validation_loss_fal.reset_states()

        with self.comet_logger.test():
            for (x_batch, y_batch) in self.data.validation_data_winter:
                predictions = self.model(x_batch)
                loss = self.loss_object(y_batch, predictions)
                self.validation_pit_hist_win.update_hist(y_batch, predictions)
                self.validation_loss_win(loss)
            self.comet_logger.log_histogram_3d(
                name="pit_histogram_win", values=self.validation_pit_hist_win.result()
            )
            self.comet_logger.log_asset_data(data=self.validation_pit_hist_win.result_as_json(),
                                        file_name="pit_histogram_win_data" + str(self.optimizer.iterations.numpy()))

            self.comet_logger.log_metric(
                "average_loss_win", self.validation_loss_win.result(), step=self.optimizer.iterations
            )
            for (x_batch, y_batch) in self.data.validation_data_spring:
                predictions = self.model(x_batch)
                loss = self.loss_object(y_batch, predictions)
                self.validation_pit_hist_spr.update_hist(y_batch, predictions)
                self.validation_loss_spr(loss)
            self.comet_logger.log_histogram_3d(
                name="pit_histogram_spr", values=self.validation_pit_hist_spr.result()
            )
            self.comet_logger.log_asset_data(data=self.validation_pit_hist_spr.result_as_json(),
                                        file_name="pit_histogram_spr_data" + str(self.optimizer.iterations.numpy()))
            self.comet_logger.log_metric(
                "average_loss_spr", self.validation_loss_spr.result(), step=self.optimizer.iterations
            )

            for (x_batch, y_batch) in self.data.validation_data_summer:
                predictions = self.model(x_batch)
                loss = self.loss_object(y_batch, predictions)
                self.validation_pit_hist_sum.update_hist(y_batch, predictions)
                self.validation_loss_sum(loss)
            self.comet_logger.log_histogram_3d(
                name="pit_histogram_sum", values=self.validation_pit_hist_sum.result()
            )
            self.comet_logger.log_asset_data(data=self.validation_pit_hist_sum.result_as_json(),
                                        file_name="pit_histogram_sum_data" + str(self.optimizer.iterations.numpy()))
            self.comet_logger.log_metric(
                "average_loss_sum", self.validation_loss_sum.result(), step=self.optimizer.iterations
            )

            for (x_batch, y_batch) in self.data.validation_data_autumn:
                predictions = self.model(x_batch)
                loss = self.loss_object(y_batch, predictions)
                self.validation_pit_hist_fal.update_hist(y_batch, predictions)
                self.validation_loss_fal(loss)
            self.comet_logger.log_histogram_3d(
                name="pit_histogram_fal", values=self.validation_pit_hist_fal.result()
            )
            self.comet_logger.log_asset_data(data=self.validation_pit_hist_fal.result_as_json(),
                                        file_name="pit_histogram_fal_data" + str(self.optimizer.iterations.numpy()))
            self.comet_logger.log_metric(
                "average_loss_fal", self.validation_loss_fal.result(), step=self.optimizer.iterations
            )


            if self.validation_loss_win.result() < self.best_loss:
                self.best_loss = self.validation_loss_win.result()
                model_files = list_files_in_directory(self.config.checkpoint_dir)
                self.save_model()
                for f in model_files:
                    os.remove(f)


    def evaluate_ensemble_crps(self):
         with self.comet_logger.test():
            for (x_batch, y_batch) in self.data.validation_data_winter:
                predictions = self.model(x_batch)
                ensemble_loss = self.ensemble_loss(y_batch, predictions)
                self.validation_ensemble_loss_win(ensemble_loss)

            self.comet_logger.log_metric(
                "ensemble_loss_win", self.validation_ensemble_loss_win.result(), step=self.optimizer.iterations
            )
            for (x_batch, y_batch) in self.data.validation_data_spring:
                predictions = self.model(x_batch)
                ensemble_loss = self.ensemble_loss(y_batch, predictions)
                self.validation_ensemble_loss_spr(ensemble_loss)

            self.comet_logger.log_metric(
                "ensemble_loss_spr", self.validation_ensemble_loss_spr.result(), step=self.optimizer.iterations
            )

            for (x_batch, y_batch) in self.data.validation_data_summer:
                predictions = self.model(x_batch)
                ensemble_loss = self.ensemble_loss(y_batch, predictions)
                self.validation_ensemble_loss_sum(ensemble_loss)

            self.comet_logger.log_metric(
                "ensemble_loss_sum", self.validation_ensemble_loss_sum.result(), step=self.optimizer.iterations
            )

            for (x_batch, y_batch) in self.data.validation_data_autumn:
                predictions = self.model(x_batch)
                ensemble_loss = self.ensemble_loss(y_batch, predictions)
                self.validation_ensemble_loss_fal(ensemble_loss)

            self.comet_logger.log_metric(
                "ensemble_loss_fal", self.validation_ensemble_loss_fal.result(), step=self.optimizer.iterations
            )


    def save_model(self):
        print("Now saving model")
        tf.saved_model.save(self.model,"7_21_after_200/")

