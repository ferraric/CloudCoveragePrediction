import xarray as xr
import numpy as np


def transform_nwp_data(data: xr.core.dataset.Dataset) -> xr.core.dataset.Dataset:
    """
    Transforms nwp data.
    Warning: this function directly modifies the input data

    Args:
        data:     xarray dataset to be transformed

    """
    data = __crop_to_fit_label_grid(data)
    try:
        data.CLCT = __clip_clct_data(data.CLCT)
    except AttributeError:
        pass
    return data


def __clip_clct_data(clct_data: xr.core.dataarray.DataArray):
    clct_data.values = np.clip(clct_data.values, 0, 100)
    return clct_data


def __crop_to_fit_label_grid(data: xr.core.dataset.Dataset) -> xr.core.dataset.Dataset:
    """
    From scripts/nwp_crop_test.py we learn that for the nwp data the following needs to hold
    in order for it to be completely within the label grid:
    -2.99 <= x <= 0.65
    -1.39 <= y <= 0.98

    Args:
        data:     xarray dataset to be cropped
    """
    return data.sel(x_1=slice(-2.99, 0.65), y_1=slice(-1.39, 0.98))
