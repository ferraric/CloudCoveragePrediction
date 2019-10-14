import numpy as np
import pandas as pd
import xarray as xr
import properscoring as ps
from typing import Tuple, Dict
import os
import pickle
import datetime
import time

with open('../../shared_data/lookup_xy_ll.pkl', 'rb') as handle:
    label_map_dict_xy_ll = pickle.load(handle)
label_map_dict_xy_ll


def findNearestLabelCoordinates(x_coord: float, y_coord: float) -> Tuple[float, float]:
    return label_map_dict_xy_ll[(x_coord, y_coord)]


predictions = xr.open_mfdataset(
    "../../../../../mnt/ds3lab-scratch/bhendj/grids/cosmo/cosmoe/cosmo-e_2018122500_CLCT.nc", combine='by_coords')
clct_predictions = predictions.CLCT
clct_predictions
yx_to_ll_df = clct_predictions.to_dataframe().reset_index(["time", "epsd_1"])[['lon_1', 'lat_1']]
yx_to_ll_dict = dict(zip(yx_to_ll_df.index.values, yx_to_ll_df.values))
yx_to_ll_dict
labels_dir = "../../local_data/labels"
labels = xr.open_mfdataset(os.path.join(labels_dir,
                                        "../../../../../mnt/ds3lab-scratch/bhendj/grids/CM-SAF/meteosat.CFC.H_ch05.latitude_longitude_201812*.nc"),
                           combine='by_coords').CFC
labels
errors = {"lat": [], "lon": [], "init_time": [], "time": [], "CRPS": []}
init_time = pd.Timestamp(2018, 12, 25, 0)
prediction_data = clct_predictions.values
for idx_t in range(prediction_data.shape[0]):
    eval_time = init_time + pd.Timedelta(hours=idx_t)
    print(datetime.datetime.now())
    print(idx_t)
    for idx_x in range(prediction_data.shape[1]):
        for idx_y in range(prediction_data.shape[2]):
            t = clct_predictions.time.values[idx_t]
            x = clct_predictions.x_1.values[idx_x]
            y = clct_predictions.y_1.values[idx_y]
            lon, lat = yx_to_ll_dict[(y, x)]
            nearestLabelCoordinates = findNearestLabelCoordinates(x, y)
            labelValue = labels.sel(lat=nearestLabelCoordinates[1], lon=nearestLabelCoordinates[0], time=t).values
            predictedValues = prediction_data[idx_t, :, idx_y, idx_x]
            error = ps.crps_ensemble(labelValue, predictedValues)
            errors["lat"].append(lat)
            errors["lon"].append(lon)
            errors["init_time"].append(init_time)
            errors["time"].append(eval_time)
            errors["CRPS"].append(error)

f = open("errors2018122500.pkl", "wb")
pickle.dump(errors, f)
f.close()
