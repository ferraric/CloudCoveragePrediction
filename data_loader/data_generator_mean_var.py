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
        #path_labels="/mnt/ds3lab-scratch/srhea/code/ml-pipeline/data_loader/subset_mean_var.pkl"
        path_labels="/mnt/ds3lab-scratch/srhea/code/input_to_tf_labels.pkl"
        full_data=pd.read_pickle(path_labels)
        CLCT_mean=np.array(full_data["CLCT_mean"],dtype="float64")
        CLCT_var=np.array(full_data["CLCT_var"],dtype="float64")
        labels=np.array(full_data["labels"],dtype="float64")
        months=np.array(full_data["month"],dtype="int32")
        hours=np.array(full_data["hour"],dtype="float64")
        year=np.array(full_data["year"],dtype="float64")
        init_time=np.array(full_data["init_time"],dtype="float64")
        lat_lon_id=np.array(full_data["lat_lon_id"],dtype="float64")
        lead_time=np.array(full_data["lead_time"],dtype="float64")
        del full_data
        CLCT_mean_train=CLCT_mean[year<=2016]
        CLCT_var_train=CLCT_var[year<=2016]
        labels_train=labels[year<=2016]
        months_train=months[year<=2016]
        hours_train=hours[year<=2016]
        year_train=year[year<=2016]
        lat_lon_id_train=lat_lon_id[year<=2016]
        init_time_train=init_time[year<=2016]
        lead_time_train=lead_time[year<=2016]
        CLCT_mean_train=np.reshape(CLCT_mean_train,[np.shape(CLCT_mean_train)[0],1])
        CLCT_var_train=np.reshape(CLCT_var_train,[np.shape(CLCT_var_train)[0],1])
        months_train=np.reshape(months_train,[np.shape(months_train)[0],1])
        hours_train=np.reshape(hours_train,[np.shape(hours_train)[0],1])
        lat_lon_id_train=np.reshape(lat_lon_id_train,[np.shape(year_train)[0],1])
        init_time_train=np.reshape(init_time_train,[np.shape(init_time_train)[0],1])
        lead_time_train=np.reshape(lead_time_train,[np.shape(lead_time_train)[0],1])
        inputs_train=np.concatenate([CLCT_mean_train,CLCT_var_train,months_train,hours_train,init_time_train,lat_lon_id_train,lead_time_train],axis=1)
        assert not np.any(np.isnan(inputs_train))
        assert not np.any(np.isnan(labels_train))
        p = np.random.permutation(len(inputs_train))
        inputs_train=inputs_train[p,:]
        labels_train=labels_train[p]
        self.train_data = tf.data.Dataset.from_tensor_slices((inputs_train, labels_train))
        self.train_data = self.train_data.shuffle(
            buffer_size=self.config.shuffle_buffer_size
        )
        self.train_data = self.train_data.batch(
            self.config.batch_size, drop_remainder=True
        )

        CLCT_mean_val_winter=CLCT_mean[year==2017*np.isin(months,[12,1,2])]
        CLCT_var_val_winter=CLCT_var[year==2017*np.isin(months,[12,1,2])]
        CLCT_mean_val_winter=np.reshape(CLCT_mean_val_winter,[np.shape(CLCT_mean_val_winter)[0],1])
        CLCT_var_val_winter=np.reshape(CLCT_var_val_winter,[np.shape(CLCT_var_val_winter)[0],1])
        labels_val_winter=labels[year==2017*np.isin(months,[12,1,2])]
        months_val_winter=months[year==2017*np.isin(months,[12,1,2])]
        hours_val_winter=hours[year==2017*np.isin(months,[12,1,2])]
        lat_lon_id_val_winter=lat_lon_id[year==2017*np.isin(months,[12,1,2])]
        init_time_val_winter=init_time[year==2017*np.isin(months,[12,1,2])]
        lead_time_val_winter=lead_time[year==2017*np.isin(months,[12,1,2])]
        print(np.max(lead_time_val_winter)) 
        months_val_winter=np.reshape(months_val_winter,[np.shape(months_val_winter)[0],1])
        hours_val_winter=np.reshape(hours_val_winter,[np.shape(hours_val_winter)[0],1])
        init_time_val_winter=np.reshape(init_time_val_winter,[np.shape(init_time_val_winter)[0],1])
        lead_time_val_winter=np.reshape(lead_time_val_winter,[np.shape(lead_time_val_winter)[0],1])
        lat_lon_id_val_winter=np.reshape(lat_lon_id_val_winter,[np.shape(lat_lon_id_val_winter)[0],1])
        validation_labels = labels_val_winter
        validation_inputs = np.concatenate([CLCT_mean_val_winter,CLCT_var_val_winter,months_val_winter,hours_val_winter,init_time_val_winter,lat_lon_id_val_winter,lead_time_val_winter],axis=1)
        self.validation_data_winter= tf.data.Dataset.from_tensor_slices((validation_inputs, validation_labels))
        self.validation_data_winter = self.validation_data_winter.batch(self.config.batch_size)
        CLCT_mean_val_spring=CLCT_mean[year==2017*np.isin(months,[3,4,5])]
        CLCT_var_val_spring=CLCT_var[year==2017*np.isin(months,[3,4,5])]
        CLCT_mean_val_spring=np.reshape(CLCT_mean_val_spring,[np.shape(CLCT_mean_val_spring)[0],1])
        CLCT_var_val_spring=np.reshape(CLCT_var_val_spring,[np.shape(CLCT_var_val_spring)[0],1])
        labels_val_spring=labels[year==2017*np.isin(months,[3,4,5])]
        months_val_spring=months[year==2017*np.isin(months,[3,4,5])]
        hours_val_spring=hours[year==2017*np.isin(months,[3,4,5])]
        lat_lon_id_val_spring=lat_lon_id[year==2017*np.isin(months,[3,4,5])]
        init_time_val_spring=init_time[year==2017*np.isin(months,[3,4,5])]
        lead_time_val_spring=lead_time[year==2017*np.isin(months,[3,4,5])]
        print(np.max(lead_time_val_spring))
        months_val_spring=np.reshape(months_val_spring,[np.shape(months_val_spring)[0],1])
        hours_val_spring=np.reshape(hours_val_spring,[np.shape(hours_val_spring)[0],1])
        init_time_val_spring=np.reshape(init_time_val_spring,[np.shape(init_time_val_spring)[0],1])
        lead_time_val_spring=np.reshape(lead_time_val_spring,[np.shape(init_time_val_spring)[0],1])
        lat_lon_id_val_spring=np.reshape(lat_lon_id_val_spring,[np.shape(lat_lon_id_val_spring)[0],1])
        validation_labels = labels_val_spring
        validation_inputs = np.concatenate([CLCT_mean_val_spring,CLCT_var_val_spring,months_val_spring,hours_val_spring,init_time_val_spring,lat_lon_id_val_spring,lead_time_val_spring],axis=1)
        self.validation_data_spring= tf.data.Dataset.from_tensor_slices((validation_inputs, validation_labels))
        self.validation_data_spring = self.validation_data_spring.batch(self.config.batch_size)
        CLCT_mean_val_summer=CLCT_mean[year==2017*np.isin(months,[6,7,8])]
        CLCT_var_val_summer=CLCT_var[year==2017*np.isin(months,[6,7,8])]
        CLCT_mean_val_summer=np.reshape(CLCT_mean_val_summer,[np.shape(CLCT_mean_val_summer)[0],1])
        CLCT_var_val_summer=np.reshape(CLCT_var_val_summer,[np.shape(CLCT_var_val_summer)[0],1])
        labels_val_summer=labels[year==2017*np.isin(months,[6,7,8])]
        months_val_summer=months[year==2017*np.isin(months,[6,7,8])]
        hours_val_summer=hours[year==2017*np.isin(months,[6,7,8])]
        lat_lon_id_val_summer=lat_lon_id[year==2017*np.isin(months,[6,7,8])]
        init_time_val_summer=init_time[year==2017*np.isin(months,[6,7,8])]
        lead_time_val_summer=lead_time[year==2017*np.isin(months,[6,7,8])]
        print(np.max(lead_time_val_summer))
        months_val_summer=np.reshape(months_val_summer,[np.shape(months_val_summer)[0],1])
        hours_val_summer=np.reshape(hours_val_summer,[np.shape(hours_val_summer)[0],1])
        lead_time_val_summer=np.reshape(lead_time_val_summer,[np.shape(lead_time_val_summer)[0],1])
        init_time_val_summer=np.reshape(init_time_val_summer,[np.shape(init_time_val_summer)[0],1])
        lat_lon_id_val_summer=np.reshape(lat_lon_id_val_summer,[np.shape(lat_lon_id_val_summer)[0],1])
        validation_labels = labels_val_summer
        validation_inputs = np.concatenate([CLCT_mean_val_summer,CLCT_var_val_summer,months_val_summer,hours_val_summer,init_time_val_summer,lat_lon_id_val_summer,lead_time_val_summer],axis=1)
        self.validation_data_summer= tf.data.Dataset.from_tensor_slices((validation_inputs, validation_labels))
        self.validation_data_summer = self.validation_data_summer.batch(self.config.batch_size)
        CLCT_mean_val_autumn=CLCT_mean[year==2017*np.isin(months,[9,10,11])]
        CLCT_var_val_autumn=CLCT_var[year==2017*np.isin(months,[9,10,11])]
        CLCT_mean_val_autumn=np.reshape(CLCT_mean_val_autumn,[np.shape(CLCT_mean_val_autumn)[0],1])
        CLCT_var_val_autumn=np.reshape(CLCT_var_val_autumn,[np.shape(CLCT_var_val_autumn)[0],1])
        labels_val_autumn=labels[year==2017*np.isin(months,[9,10,11])]
        months_val_autumn=months[year==2017*np.isin(months,[9,10,11])]
        hours_val_autumn=hours[year==2017*np.isin(months,[9,10,11])]
        lat_lon_id_val_autumn=lat_lon_id[year==2017*np.isin(months,[9,10,11])]
        lead_time_val_autumn=lead_time[year==2017*np.isin(months,[9,10,11])]
        print(np.max(lead_time_val_autumn))
        init_time_val_autumn=init_time[year==2017*np.isin(months,[9,10,11])]
        months_val_autumn=np.reshape(months_val_autumn,[np.shape(months_val_autumn)[0],1])
        hours_val_autumn=np.reshape(hours_val_autumn,[np.shape(hours_val_autumn)[0],1])
        init_time_val_autumn=np.reshape(init_time_val_autumn,[np.shape(init_time_val_autumn)[0],1])
        lead_time_val_autumn=np.reshape(lead_time_val_autumn,[np.shape(init_time_val_autumn)[0],1])
        lat_lon_id_val_autumn=np.reshape(lat_lon_id_val_autumn,[np.shape(lat_lon_id_val_autumn)[0],1])
        validation_labels = labels_val_autumn
        validation_inputs = np.concatenate([CLCT_mean_val_autumn,CLCT_var_val_autumn,months_val_autumn,hours_val_autumn,init_time_val_autumn,lat_lon_id_val_autumn,lead_time_val_autumn],axis=1)
        self.validation_data_autumn= tf.data.Dataset.from_tensor_slices((validation_inputs, validation_labels))
        self.validation_data_autumn= self.validation_data_autumn.batch(self.config.batch_size)
        self.comet_logger.log_dataset_hash(self.train_data)
        self.comet_logger.log_dataset_hash(self.validation_data_winter)
        self.comet_logger.log_dataset_hash(self.validation_data_summer)
        self.comet_logger.log_dataset_hash(self.validation_data_spring)
        self.comet_logger.log_dataset_hash(self.validation_data_autumn)
