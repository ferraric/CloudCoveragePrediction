import numpy as np
import pandas as pd
import xarray as xr
import properscoring as ps
from typing import Tuple
import pickle
import time
import itertools
import os
import sys

sys.path.append('../preprocessing/')
from nwp_preprocessing import transform_nwp_data

with open('../../shared_data/lookup_xy_ll.pkl', 'rb') as handle:
    label_map_dict_xy_ll = pickle.load(handle)
with open('../../shared_data/yx_to_ll_dict.pkl', 'rb') as handle:
    yx_to_ll_dict = pickle.load(handle)

with open("bias_dict.pkl","rb") as f:
     dictionary=pickle.load(f)
def findNearestLabelCoordinates(x_coord: float, y_coord: float) -> Tuple[float, float]:
    return label_map_dict_xy_ll[(x_coord, y_coord)]


# opening all label files since they are small
labels = xr.open_mfdataset(
    "../../../../../mnt/ds3lab-scratch/bhendj/grids/CM-SAF/MeteosatCFC/meteosat.CFC.H_ch05.latitude_longitude_201406*.nc",
    combine='by_coords').CFC
longitudes_latitudes_labels = list(itertools.product(list(labels.lon.values), list(labels.lat.values)))

# open arbitrary nwp clct file to obtain coordinates
predictions = xr.open_mfdataset(
    "../../../../../mnt/ds3lab-scratch/bhendj/grids/cosmo/cosmoe/cosmo-e_2018122500_CLCT.nc")
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

start_date = pd.Timestamp(2014, 1, 1, 0)
end_date = pd.Timestamp(2018, 12, 31, 12)  # pd.Timestamp(2018, 12, 31, 12)
time_step = pd.Timedelta(hours=12)
init_time = start_date
missing_dates = []
while init_time <= end_date:
    print("initialization time: ", init_time.strftime("%Y-%m-%d-%H"))

    if os.path.exists("nwp_crps_scores" + init_time.strftime("%Y-%m-%d-%H") + ".pkl"):
        print("file already exists, skipping to next initialization time")
        init_time += time_step
        continue
    init_time_asint=int(init_time.hour)

    try:
        predictions = xr.open_mfdataset("../../../../../mnt/ds3lab-scratch/bhendj/grids/cosmo/cosmoe/cosmo-e_*" +
                                        init_time.strftime("%Y%m%d%H") + "*_CLCT.nc")
        predictions = transform_nwp_data(predictions)
        clct_predictions = predictions.CLCT
        prediction_data = clct_predictions.values
    except:
        missing_dates.append(init_time)
        init_time += time_step
        continue

    errors = {"lat": [], "lon": [], "init_time": [], "time": [], "CRPS": []}


    for idx_t in range(prediction_data.shape[0]):
        start_timer = time.time()
        eval_time = init_time + pd.Timedelta(hours=idx_t)
        lead_time=(eval_time-init_time).astype('timedelta64[h]')
        month=eval_time.month
        print(idx_t)
        t = clct_predictions.time.values[idx_t]
        all_labels_values = list(labels.sel(time=t).values.flatten(order='F'))
        label_coord_to_value_dict = dict(zip(longitudes_latitudes_labels, all_labels_values))
        keys = [(lat_array[i], lon_array[i], init_time_asint, lead_time,month) for i in range(0, len(lat_array))]
        label_values_for_nwp_grid_points = np.zeros(nearestLabelCoordinates.shape[0])
        i = 0
        for lon_n, lat_n in nearestLabelCoordinates:
            label_values_for_nwp_grid_points[i] = label_coord_to_value_dict[(lon_n, lat_n)]
            i += 1
        predicted_values = prediction_data[idx_t, :, idx_x_for_each_grid_point, idx_y_for_each_grid_point]
        bias=np.array([dictionary[x] for x in keys])
        crps_scores = ps.crps_ensemble(label_values_for_nwp_grid_points, predicted_values-bias)
        assert (crps_scores.shape == lat_array.shape and crps_scores.shape == lon_array.shape)
        errors["lat"].append(lat_array)
        errors["lon"].append(lon_array)
        errors["init_time"].append(np.repeat(init_time, crps_scores.shape[0]))
        errors["time"].append(np.repeat(eval_time, crps_scores.shape[0]))
        errors["CRPS"].append(crps_scores)
        f = open("nwp_crps_scores" + init_time.strftime("%Y-%m-%d-%H") + ".pkl", "wb")
        pickle.dump(errors, f)
        f.close()
        print("--- %s seconds ---" % (time.time() - start_timer))
    init_time += time_step

f = open("missing_dates" + start_date.strftime("%Y-%m-%d-%H") + "_to_" + end_date.strftime("%Y-%m-%d-%H") + ".pkl",
         "wb")
pickle.dump(missing_dates, f)
f.close()

