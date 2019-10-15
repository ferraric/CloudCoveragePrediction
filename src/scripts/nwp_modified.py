def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a
import numpy as np
import pandas as pd
import xarray as xr
import properscoring as ps
from typing import Tuple, Dict
import os
import pickle
import datetime
import time

with open('lookup_xy_ll.pkl', 'rb') as handle:
    label_map_dict_xy_ll = pickle.load(handle)
label_map_dict_xy_ll


def findNearestLabelCoordinates(x_coord: float, y_coord: float) -> Tuple[float, float]:
    return label_map_dict_xy_ll[(x_coord, y_coord)]


predictions = xr.open_mfdataset(
    "cosmo-e_2018122500_CLCT.nc", combine='by_coords')
clct_predictions = predictions.CLCT
clct_predictions
yx_to_ll_df = clct_predictions.to_dataframe().reset_index(["time", "epsd_1"])[['lon_1', 'lat_1']]
yx_to_ll_dict = dict(zip(yx_to_ll_df.index.values, yx_to_ll_df.values))
yx_to_ll_dict
labels = xr.open_mfdataset(os.path.join("meteosat.CFC.H_ch05.latitude_longitude_201812*.nc"),
                           combine='by_coords').CFC
labels
errors = {"lat": [], "lon": [], "init_time": [], "time": [], "CRPS": []}
prediction_data = clct_predictions.values
init_time = pd.Timestamp(2018, 12, 25, 0)
prediction_data = clct_predictions.values
for idx_t in range(prediction_data.shape[0]):
    #import time
    start_time = time.time()
    eval_time = init_time + pd.Timedelta(hours=idx_t)
    print(datetime.datetime.now())
    print(idx_t)
    print(eval_time)
    t = clct_predictions.time.values[idx_t]
    x = clct_predictions.x_1.values
    y = clct_predictions.y_1.values
    a = np.repeat(x, y.shape[0], axis=0)
    b=np.tile(y,x.shape[0]) 
    result = np.vstack([b,a])
    result=np.transpose(result)
    tup_ind=totuple(result)
    latlonarr = np.array([yx_to_ll_dict[(h,j)] for h,j in tup_ind ])
    lon=latlonarr[:,0]
    lat=latlonarr[:,1]
    res2 = np.vstack([a,b])
    res2=totuple(np.transpose(res2))
    nearestLabelCoordinates = [findNearestLabelCoordinates(x, y) for (x,y) in res2]
    nearestLabelCoordinates =np.array(nearestLabelCoordinates)
    labelValue = labels.sel(lat=nearestLabelCoordinates[:,1], lon=nearestLabelCoordinates[:,0], time=t).values
    labelValue=np.diag(labelValue)
    idx_y=list(range(0,prediction_data.shape[3]))
    idx_x=list(range(0,prediction_data.shape[2]))
    idy=np.repeat(idx_y,np.shape(idx_x)[0])
    idx=np.tile(idx_x,np.shape(idx_y)[0])
    predictedValues = prediction_data[idx_t, :,idx, idy]
    err = ps.crps_ensemble(labelValue, predictedValues)
    #print(predictedValues.shape)
    #error = ps.crps_ensemble(labelValue, predictedValues)
    #errors["lat"].append(lat)
    #errors["lon"].append(lon)
    #errors["init_time"].append(init_time)
    #errors["time"].append(eval_time)
    #errors["CRPS"].append(error)
    print("--- %s seconds ---" % (time.time() - start_time))
    #break
#f = open("errors2018122500.pkl", "wb")
#pickle.dump(errors, f)
#f.close()
