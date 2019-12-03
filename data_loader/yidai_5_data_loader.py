import numpy as np
import tensorflow as tf
import xarray as xr
import pickle
import pandas as pd
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple
import pickle
import time
import itertools
import os
import sys
class DataGenerator:
    def printer(self, string):
        print("Reading feature", string)
        return string


    def __init__(self, config, comet_logger):
        self.config = config
        self.comet_logger = comet_logger

        assert (
            "batch_size" in self.config
        ), "You need to define the parameter 'batch_size' in your config file."
        assert (
            "shuffle_buffer_size" in self.config
        ), "You need to define the parameter 'shuffle_buffer_size' in your config file."
        #full_data=pd.read_pickle("/mnt/ds3lab-scratch/yidai/nwp_subsampled_2x2_5_day_stride_ensemble_mean_var/input_to_tf_biases_winter_it00.pkl")
        folder = "/mnt/ds3lab-scratch/srhea/all_columns/"
        feature_names = ["month", "hour", "init_time", "lat_lon_id"]
        feature_names.extend(["quantiles_" + str(q) for q in range(21)])
        labels = np.array(pd.read_pickle(folder + "labels.pkl"), dtype="float64").flatten()
        print("Successfully read labels")
        finite_ind = np.isfinite(labels)
        labels = labels[finite_ind]
        features = [np.array(pd.read_pickle(folder + self.printer(name) + ".pkl"), dtype="float64").flatten()[finite_ind] for name in feature_names]
        print("Successfully read all features")
        years = np.array(pd.read_pickle(folder + "year.pkl"), dtype="float64").flatten()[finite_ind]
        print("Successfully read years")
        train_ind = years <= 2016
        test_ind_win = np.logical_and(years == 2017, np.logical_or(features[0] == 12, features[0] <= 2))
        test_ind_spr = np.logical_and(years == 2017, np.logical_and(features[0] >= 3, features[0] <= 5))
        test_ind_sum = np.logical_and(years == 2017, np.logical_and(features[0] >= 6, features[0] <= 8))
        test_ind_fal = np.logical_and(years == 2017, np.logical_and(features[0] >= 9, features[0] <= 11)) 
        del years
        features_train = [feature[train_ind].reshape((-1, 1)) for feature in features]
        features_test_win = [feature[test_ind_win].reshape((-1, 1)) for feature in features]
        features_test_spr = [feature[test_ind_spr].reshape((-1, 1)) for feature in features]
        features_test_sum = [feature[test_ind_sum].reshape((-1, 1)) for feature in features]
        features_test_fal = [feature[test_ind_fal].reshape((-1, 1)) for feature in features]

        del features
        labels_train = labels[train_ind]
        labels_test_win = labels[test_ind_win]
        labels_test_spr = labels[test_ind_spr]
        labels_test_sum = labels[test_ind_sum]
        labels_test_fal = labels[test_ind_fal]
        del labels
        #year_train=year[year<=2016]
        #CLCT_mean_train=np.reshape(CLCT_mean_train,[np.shape(CLCT_mean_train)[0],1])
        #CLCT_var_train=np.reshape(CLCT_var_train,[np.shape(CLCT_var_train)[0],1])
        #months_train=np.reshape(months_train,[np.shape(months_train)[0],1])
        #hours_train=np.reshape(hours_train,[np.shape(hours_train)[0],1])
        #lat_lon_id_train=np.reshape(lat_lon_id_train,[np.shape(year_train)[0],1])
        #init_time_train=np.reshape(init_time_train,[np.shape(init_time_train)[0],1])
        inputs_train = np.concatenate(features_train, axis=1)
        del features_train
        assert not np.any(np.isnan(inputs_train))
        print(labels_train)
        print(np.shape(labels_train))
        print(np.shape(inputs_train))
        assert not np.any(np.isnan(labels_train))
        p = np.random.permutation(len(inputs_train))
        inputs_train=inputs_train[p,:]
        labels_train=labels_train[p]
        self.train_data = tf.data.Dataset.from_tensor_slices((inputs_train, labels_train))
        self.train_data = self.train_data.shuffle(
            buffer_size=self.config.shuffle_buffer_size
        ).repeat(100)
        self.train_data = self.train_data.batch(
            self.config.batch_size, drop_remainder=True
        )

        # dummy input
        #CLCT_mean_val=np.reshape(CLCT_mean_val,[np.shape(CLCT_mean_val)[0],1])
        #CLCT_var_val=np.reshape(CLCT_var_val,[np.shape(CLCT_var_val)[0],1])
        #months_val=np.reshape(months_val,[np.shape(months_val)[0],1])
        #hours_val=np.reshape(hours_val,[np.shape(hours_val)[0],1])
        #init_time_val=np.reshape(init_time_val,[np.shape(init_time_val)[0],1])
        #lat_lon_id_val=np.reshape(lat_lon_id_val,[np.shape(lat_lon_id_val)[0],1])
        validation_inputs_win = np.concatenate(features_test_win, axis=1)
        validation_inputs_spr = np.concatenate(features_test_spr, axis=1)
        validation_inputs_sum = np.concatenate(features_test_sum, axis=1)
        validation_inputs_fal = np.concatenate(features_test_fal, axis=1)
        del features_test_win, features_test_spr, features_test_sum, features_test_fal
        validation_labels_win = labels_test_win
        validation_labels_spr = labels_test_spr
        validation_labels_sum = labels_test_sum
        validation_labels_fal = labels_test_fal
        self.validation_data_win = tf.data.Dataset.from_tensor_slices((validation_inputs_win, validation_labels_win)).batch(self.config.batch_size)
        self.validation_data_spr = tf.data.Dataset.from_tensor_slices((validation_inputs_spr, validation_labels_spr)).batch(self.config.batch_size)
        self.validation_data_sum = tf.data.Dataset.from_tensor_slices((validation_inputs_sum, validation_labels_sum)).batch(self.config.batch_size)
        self.validation_data_fal = tf.data.Dataset.from_tensor_slices((validation_inputs_fal, validation_labels_fal)).batch(self.config.batch_size)
#        self.validation_data = self.validation_data.batch(self.config.batch_size)
        self.comet_logger.log_dataset_hash(self.train_data)
        self.comet_logger.log_dataset_hash(self.validation_data_win)
        self.comet_logger.log_dataset_hash(self.validation_data_spr)
        self.comet_logger.log_dataset_hash(self.validation_data_sum)
        self.comet_logger.log_dataset_hash(self.validation_data_fal)

