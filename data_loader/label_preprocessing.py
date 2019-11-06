import xarray as xr
import numpy as np
import pickle
import itertools

from typing import Tuple


with open('../shared_data/lookup_xy_ll.pkl', 'rb') as handle:
    label_map_dict_xy_ll = pickle.load(handle)
with open('../shared_data/yx_to_ll_dict.pkl', 'rb') as handle:
    yx_to_ll_dict = pickle.load(handle)


def findNearestLabelCoordinates(x_coord: float, y_coord: float) -> Tuple[float, float]:
    return label_map_dict_xy_ll[(x_coord, y_coord)]


def map_to_nwp_grid(data: xr.core.dataset.Dataset, nwp_example: xr.core.dataset.Dataset) -> xr.core.dataset.Dataset:
    print(data)

    transformed_labels = nwp_example.copy(deep=True)
    desired_shape = transformed_labels.values.shape
    transformed_labels.values = np.zeros(desired_shape)

    x_coordinates = transformed_labels.x_1.values
    y_coordinates = transformed_labels.y_1.values
    x_coordinates_for_each_grid_point = np.repeat(x_coordinates, y_coordinates.shape[0], axis=0)
    y_coordinates_for_each_grid_point = np.tile(y_coordinates, x_coordinates.shape[0])
    yx_coordinates_for_each_grid_point = np.vstack(
        [y_coordinates_for_each_grid_point, x_coordinates_for_each_grid_point])
    yx_coordinates_for_each_grid_point = np.transpose(yx_coordinates_for_each_grid_point)
    nearestLabelCoordinates = [findNearestLabelCoordinates(x, y) for (y, x) in yx_coordinates_for_each_grid_point]
    nearestLabelCoordinates = np.array(nearestLabelCoordinates)

    longitudes_latitudes_labels = list(itertools.product(data.lon.values, data.lat.values))
    all_labels_values = list(data.CFC.isel(time=0).values.flatten(order='F'))
    label_coord_to_value_dict = dict(zip(longitudes_latitudes_labels, all_labels_values))
    new_label_values = np.reshape(
        np.array([label_coord_to_value_dict[(lon, lat)] for (lon, lat) in nearestLabelCoordinates]), desired_shape)
    transformed_labels.values = new_label_values
    res = transformed_labels.expand_dims(dim='time').assign_coords(time=data.time.values)
    print(res)
    return res
