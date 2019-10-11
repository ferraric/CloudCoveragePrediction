import xarray as xr
import numpy as np


def transform_nwp_data(data: xr.core.dataset.Dataset):
    """
    Transforms nwp data.
    Warning: this function directly modifies the input data

    Args:
        data:     xarray dataset to be transformed

    """
    try:
        __clip_clct_data(data.CLCT)
    except AttributeError:
        pass


def __clip_clct_data(clct_data: xr.core.dataarray.DataArray):
    clct_data.values = np.clip(clct_data.values, 0, 100)
    return clct_data
