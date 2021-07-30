import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple
import pickle
import time
import itertools
import os
import sys

sys.path.append('../data_loader/')
from nwp_preprocessing import transform_nwp_data

with open('../shared_data/lookup_xy_ll.pkl', 'rb') as handle:
    label_map_dict_xy_ll = pickle.load(handle)
with open('../shared_data/yx_to_ll_dict.pkl', 'rb') as handle:
    yx_to_ll_dict = pickle.load(handle)


def findNearestLabelCoordinates(x_coord: float, y_coord: float) -> Tuple[float, float]:
    return label_map_dict_xy_ll[(x_coord, y_coord)]

def pit_hist(y_true, y_pred):
    assert y_true.shape[0] == y_pred.shape[0]
    y_true = np.reshape(y_true, (y_true.shape[0], 1))
    all_values = np.append(y_pred, y_true, axis=-1)
    corner_cases_ind = ((all_values == 0.0) | (all_values == 100.0))
    all_values[corner_cases_ind] = all_values[corner_cases_ind] + 0.001 * np.random.standard_normal(all_values[corner_cases_ind].shape)
    ranks = all_values.argsort(axis=-1).argsort(axis=-1)
    return  ranks[:,-1]

save_path = "/mnt/ds3lab-scratch/ferraric/pit_hist_ranks"
# opening all label files since they are small
labels = xr.open_mfdataset(
    "/mnt/ds3lab-scratch/bhendj/grids/CM-SAF/MeteosatCFC/meteosat.CFC.H_ch05.latitude_longitude_201[8-8]*.nc",
    combine='by_coords').CFC
longitudes_latitudes_labels = list(itertools.product(labels.lon.values, labels.lat.values))

# open arbitrary nwp clct file to obtain coordinates
predictions = xr.open_mfdataset(
    "/mnt/ds3lab-scratch/bhendj/grids/cosmo/cosmoe/cosmo-e_2018122500_CLCT.nc")
predictions = transform_nwp_data(predictions)
clct_predictions = predictions.CLCT
prediction_data = clct_predictions.values
x_coordinates = clct_predictions.x_1.values
y_coordinates = clct_predictions.y_1.values
x_coordinates_for_each_grid_point = np.repeat(x_coordinates, y_coordinates.shape[0], axis=0)
y_coordinates_for_each_grid_point = np.tile(y_coordinates, x_coordinates.shape[0])
yx_coordinates_for_each_grid_point = np.vstack([y_coordinates_for_each_grid_point, x_coordinates_for_each_grid_point])
yx_coordinates_for_each_grid_point = np.transpose(yx_coordinates_for_each_grid_point)
latlonarr = np.array([yx_to_ll_dict[(y, x)] for y, x in yx_coordinates_for_each_grid_point])
lon_array = latlonarr[:, 0]
lat_array = latlonarr[:, 1]
nearestLabelCoordinates = [findNearestLabelCoordinates(x, y) for (y, x) in yx_coordinates_for_each_grid_point]
nearestLabelCoordinates = np.array(nearestLabelCoordinates)
idx_y = list(range(0, prediction_data.shape[3]))
idx_x = list(range(0, prediction_data.shape[2]))
idx_y_for_each_grid_point = np.repeat(idx_y, np.shape(idx_x)[0])
idx_x_for_each_grid_point = np.tile(idx_x, np.shape(idx_y)[0])

start_date = pd.Timestamp(2018, 1, 1, 0)
end_date = pd.Timestamp(2018, 12, 31, 12)
time_step = pd.Timedelta(hours=12)
init_time = start_date
missing_dates = []
while init_time <= end_date:
    print("initialization time: ", init_time.strftime("%Y-%m-%d-%H"))

    if os.path.exists("nwp_crps_scores" + init_time.strftime("%Y-%m-%d-%H") + ".pkl"):
        print("file already exists, skipping to next initialization time")
        init_time += time_step
        continue

    try:
        predictions = xr.open_mfdataset("/mnt/ds3lab-scratch/bhendj/grids/cosmo/cosmoe/cosmo-e_*" +
                                        init_time.strftime("%Y%m%d%H") + "*_CLCT.nc", preprocess=transform_nwp_data)
        clct_predictions = predictions.CLCT
        prediction_data = clct_predictions.values
    except:
        missing_dates.append(init_time)
        init_time += time_step
        continue

    errors = {"lat": [], "lon": [], "init_time": [], "time": [], "pit_hist_rank": []}

    for idx_t in range(prediction_data.shape[0]):
        start_timer = time.time()
        eval_time = init_time + pd.Timedelta(hours=idx_t)
        print(idx_t)
        t = clct_predictions.time.values[idx_t]
        all_labels_values = list(labels.sel(time=t).values.flatten(order='F'))
        label_coord_to_value_dict = dict(zip(longitudes_latitudes_labels, all_labels_values))

        label_values_for_nwp_grid_points = np.zeros(nearestLabelCoordinates.shape[0])
        i = 0
        for lon_n, lat_n in nearestLabelCoordinates:
            label_values_for_nwp_grid_points[i] = label_coord_to_value_dict[(lon_n, lat_n)]
            i += 1
        predicted_values = prediction_data[idx_t, :, idx_x_for_each_grid_point, idx_y_for_each_grid_point]
        pit_hist_ranks = pit_hist(label_values_for_nwp_grid_points, predicted_values)
        assert (pit_hist_ranks.shape == lat_array.shape and pit_hist_ranks.shape == lon_array.shape)
        errors["lat"].extend(lat_array)
        errors["lon"].extend(lon_array)
        errors["init_time"].extend(np.repeat(init_time, pit_hist_ranks.shape[0]))
        errors["time"].extend(np.repeat(eval_time, pit_hist_ranks.shape[0]))
        errors["pit_hist_rank"].extend(pit_hist_ranks)
        print("--- %s seconds ---" % (time.time() - start_timer))
    pd.DataFrame(errors).to_pickle(os.path.join(save_path, "pit_hist_ranks" + init_time.strftime("%Y-%m-%d-%H") + ".pkl"))
    init_time += time_step

f = open("missing_dates" + start_date.strftime("%Y-%m-%d-%H") + "_to_" + end_date.strftime("%Y-%m-%d-%H") + ".pkl",
         "wb")
pickle.dump(missing_dates, f)
f.close()

