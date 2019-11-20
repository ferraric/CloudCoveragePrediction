import numpy as np
import tensorflow as tf
import xarray as xr
import pandas as pd
from data_loader.label_preprocessing import LabelTransformer

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

        nwp_data = xr.open_mfdataset(
           #"/mnt/ds3lab-scratch/ferraric/nwp_subsampled_2x2_5_day_stride_ensemble0/subsampled_CLCT_2014-01-01-00_2018-12-31-00.nc"
           #"/mnt/ds3lab-scratch/ferraric/nwp_subsampled_2x2_5_day_stride_ensemble0/subsampled_CLCT_2014-01-01-12_2018-12-31-12.nc"
            "../local/local_data/subsampled_CLCT_2014-01-01-00_2018-12-31-00.nc"
        )

        dummy_nwp = nwp_data.isel(epsd_1=0, lead_time=0, init_time=0).drop(['epsd_1', 'lead_time', 'init_time'])
        label_transformer = LabelTransformer(dummy_nwp)
        labels = xr.open_mfdataset(
            #"/mnt/ds3lab-scratch/bhendj/grids/CM-SAF/MeteosatCFC/meteosat.CFC.H_ch05.latitude_longitude_201[4-5]*.nc",
            "../local/local_data/labels/meteosat.CFC.H_ch05.latitude_longitude_201[4-4]0[2-3]*.nc",
            combine='by_coords', preprocess=label_transformer.map_to_nwp_grid)

        cutoff_date = pd.Timestamp('2014-03-30') - pd.Timedelta(hours=121)
        nwp = nwp_data.sel(init_time=slice('2014-01-01', cutoff_date)).isel(epsd_1=0).drop('epsd_1')

        label_values_per_it = np.zeros(nwp.CLCT.values.shape, dtype=np.float32)
        for i, it in enumerate(nwp.init_time.values):
            print(it)
            try:
                lead_times = nwp.lead_time.values
                label_frame = labels.sel(time=slice(it, it + pd.Timedelta(hours=lead_times[-1]))) \
                    .rename_vars({'CLCT': 'CFC'}) \
                    .assign_coords(lead_time=lead_times) \
                    .rename_dims({'time': 'lead_time'}) \
                    .expand_dims('init_time')

                label_values_per_it[i, :, :, :] = label_frame.CFC.values
            except ValueError:
                break

        nwp = nwp.assign(labelValue=(['init_time', 'lead_time', 'y_1', 'x_1'], label_values_per_it))

        val_cutoff = '2014-03-01'
        assert pd.Timestamp(val_cutoff) < pd.Timestamp(cutoff_date)
        nwp_train = nwp.sel(init_time=slice('2014-02-01',val_cutoff))
        nwp_val = nwp.sel(init_time=slice(val_cutoff,cutoff_date))

        train_values_correct_shape = np.swapaxes(nwp_train.CLCT.values, 1, 3)
        train_labels_correct_shape = np.swapaxes(nwp_train.labelValue.values, 1, 3)
        self.comet_logger.log_other("train size before removing nan", train_values_correct_shape.shape)

        nan_indices = np.any(np.isnan(train_labels_correct_shape), axis=(1, 2, 3))
        nan_it_indices = np.unique(np.argwhere(np.isnan(train_labels_correct_shape))[:, 0])
        print(nan_it_indices)
        train_values_correct_shape = train_values_correct_shape[~nan_indices, :, :, :]
        train_labels_correct_shape = train_labels_correct_shape[~nan_indices, :, :, :]
        self.comet_logger.log_other("train size after removing nan", train_values_correct_shape.shape)

        self.train_data = tf.data.Dataset.from_tensor_slices((train_values_correct_shape, train_labels_correct_shape))
        self.train_data = self.train_data.shuffle(
            buffer_size=self.config.shuffle_buffer_size
        )
        self.train_data = self.train_data.batch(self.config.batch_size)

        val_values_correct_shape = np.swapaxes(nwp_val.CLCT.values, 1, 3)
        val_labels_correct_shape = np.swapaxes(nwp_val.labelValue.values, 1, 3)
        self.comet_logger.log_other("validation size before removing nan", val_values_correct_shape.shape)

        nan_indices = np.any(np.isnan(val_labels_correct_shape), axis=(1, 2, 3))
        nan_it_indices = np.unique(np.argwhere(np.isnan(val_labels_correct_shape))[:, 0])
        print(nan_it_indices)
        val_values_correct_shape = val_values_correct_shape[~nan_indices, :, :, :]
        val_labels_correct_shape = val_labels_correct_shape[~nan_indices, :, :, :]
        self.comet_logger.log_other("validation size after removing nan", val_values_correct_shape.shape)

        self.validation_data = tf.data.Dataset.from_tensor_slices((val_values_correct_shape, val_labels_correct_shape))
        self.validation_data = self.validation_data.batch(self.config.batch_size)