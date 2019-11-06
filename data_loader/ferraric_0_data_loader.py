import numpy as np
import tensorflow as tf
import xarray as xr
from data_loader.label_preprocessing import map_to_nwp_grid

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

        nwp_data = xr.open_mfdataset("../local/local_data/subsampled_CLCT_2014-01-01-00_2018-12-31-00.nc")

        dummy_nwp = nwp_data.CLCT.isel(epsd_1=0, lead_time=0, init_time=0).drop(['epsd_1', 'lead_time', 'init_time'])
        labels = xr.open_mfdataset(
            "../local/local_data/labels/meteosat.CFC.H_ch05.latitude_longitude_201401*.nc",
            combine='by_coords', preprocess=(lambda data: map_to_nwp_grid(data, dummy_nwp)))


        print(labels)