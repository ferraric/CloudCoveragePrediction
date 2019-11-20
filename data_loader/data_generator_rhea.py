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
    def __init__(self, config, comet_logger):
        self.config = config
        self.comet_logger = comet_logger

        assert (
            "batch_size" in self.config
        ), "You need to define the parameter 'batch_size' in your config file."
        assert (
            "shuffle_buffer_size" in self.config
        ), "You need to define the parameter 'shuffle_buffer_size' in your config file."
        #self.data_subsampled = xr.open_dataset(
        #    "/mnt/ds3lab-scratch/ferraric/nwp_subsampled_2x2_5_day_stride_ensemble0/subsampled_CLCT_2014-01-01-12_2018-12-31-12.nc")
        #print("loaded subsampled")
        #self.all_init_times=self.data_subsampled.init_time.values
        #print(np.shape(self.all_init_times))
        #self.labels_upto_5_days=xr.open_mfdataset("/mnt/ds3lab-scratch/bhendj/grids/CM-SAF/MeteosatCFC/meteosat.CFC.H_ch05.latitude_longitude_201[4-5]*.nc",combine='by_coords').CFC
        #print("loaded labels")
        #inputs,labels=self.match_to_labels()
        #print(np.shape(inputs))
        #print(np.shape(labels))
        # dummy input
        #inputs = np.zeros((1, 50, 50, 1))
        #labels = np.ones((1, 50, 50, 1))
        full_data=pd.read_pickle("/mnt/ds3lab-scratch/srhea/code/ml-pipeline/data_loader/input_to_tf_2.pkl")
        CLCT_mean=np.array(full_data["CLCT_mean"],dtype="float64")
        CLCT_var=np.array(full_data["CLCT_var"],dtype="float64")
        labels=np.array(full_data["labels"],dtype="float64")
        months=np.array(full_data["month"],dtype="float64")
        hours=np.array(full_data["lead_time"],dtype="float64")
        year=np.array(full_data["year"],dtype="float64")
        init_time=np.array(full_data["init_time"],dtype="float64")
        lat_lon_id=np.array(full_data["lat_lon_id"],dtype="float64")
        #inputs=np.reshape(inputs,[np.shape(inputs)[0],1])
        #labels=np.reshape(labels,[np.shape(labels)[0],1])
        CLCT_mean_train=CLCT_mean[year<=2016]
        CLCT_mean_train=CLCT_mean_train#/ np.linalg.norm(CLCT_train)
        CLCT_var_train=CLCT_var[year<=2016]
        labels_train=labels[year<=2016]
        months_train=months[year<=2016]
        hours_train=hours[year<=2016]
        year_train=year[year<=2016]
        lat_lon_id_train=lat_lon_id[year<=2016]
        init_time_train=init_time[year<=2016]
        CLCT_mean_train=np.reshape(CLCT_mean_train,[np.shape(CLCT_mean_train)[0],1])
        CLCT_var_train=np.reshape(CLCT_var_train,[np.shape(CLCT_var_train)[0],1])
        months_train=np.reshape(months_train,[np.shape(months_train)[0],1])
        hours_train=np.reshape(hours_train,[np.shape(hours_train)[0],1])
        lat_lon_id_train=np.reshape(lat_lon_id_train,[np.shape(year_train)[0],1])
        init_time_train=np.reshape(init_time_train,[np.shape(init_time_train)[0],1])
        #labels=labels[~np.isnan(labels)]
        print(max(lat_lon_id))
        inputs_train=np.concatenate([CLCT_mean_train,CLCT_var_train,months_train,hours_train,init_time_train,lat_lon_id_train],axis=1)
        #inputs_train=inputs_train[100:200,:]
        assert not np.any(np.isnan(inputs_train))
        print(labels_train)
        print(np.shape(labels_train))
        print(np.shape(inputs_train))
        print(np.sum(np.isnan(labels_train)))
        assert not np.any(np.isnan(labels_train))
        p = np.random.permutation(len(inputs_train))
        inputs_train=inputs_train[p,:]
        labels_train=labels_train[p]
        #len_train=int(0.75*np.shape(inputs)[0])
        self.train_data = tf.data.Dataset.from_tensor_slices((inputs_train, labels_train))
        self.train_data = self.train_data.shuffle(
            buffer_size=self.config.shuffle_buffer_size
        ).repeat(100)
        self.train_data = self.train_data.batch(
            self.config.batch_size, drop_remainder=True
        )

        # dummy input
        CLCT_mean_val=CLCT_mean[year==2017]
        #CLCT_val=CLCT_val#/ np.linalg.norm(CLCT_val)
        CLCT_var_val=CLCT_var[year==2017]
        labels_val=labels[year==2017]
        months_val=months[year==2017]
        hours_val=hours[year==2017]
        lat_lon_id_val=lat_lon_id[year==2017]
        #year_val=year[(~np.isnan(labels)) & year<=2017]
        init_time_val=init_time[year==2017]
        CLCT_mean_val=np.reshape(CLCT_mean_val,[np.shape(CLCT_mean_val)[0],1])
        CLCT_var_val=np.reshape(CLCT_var_val,[np.shape(CLCT_var_val)[0],1])
        months_val=np.reshape(months_val,[np.shape(months_val)[0],1])
        hours_val=np.reshape(hours_val,[np.shape(hours_val)[0],1])
        #year_val=np.reshape(year_val,[np.shape(year_val)[0],1])
        init_time_val=np.reshape(init_time_val,[np.shape(init_time_val)[0],1])
        lat_lon_id_val=np.reshape(lat_lon_id_val,[np.shape(lat_lon_id_val)[0],1])
        validation_inputs = np.concatenate([CLCT_mean_val,CLCT_var_val,months_val,hours_val,init_time_val,lat_lon_id_val],axis=1)
        validation_labels = labels_val
        self.validation_data = tf.data.Dataset.from_tensor_slices((validation_inputs, validation_labels))
        self.validation_data = self.validation_data.batch(self.config.batch_size)
        #print(self.train_data.shape)
        self.comet_logger.log_dataset_hash(self.train_data)
        self.comet_logger.log_dataset_hash(self.validation_data)