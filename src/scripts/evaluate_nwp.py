import xarray as xr
import pandas as pd
import numpy as np
import properscoring as ps
import os
import itertools
import pickle
import warnings
import time as t
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

save_dir = os.path.join("..", "local_playground")
init_time_str = "20180221 00"
labels_dir = os.path.join("..", "local_playground")
nwp_dir = os.path.join("..", "local_playground")
lookup_dir = os.path.join("..", "..", "shared_data")
init_time = pd.Timestamp(init_time_str)

with open(os.path.join(lookup_dir, "lookup_xy_ll.pkl"), "rb") as f:
    lookup_dict = pickle.load(f)
nwp_file = os.path.join(labels_dir, "cosmo-e_" + init_time_str.replace(' ', '') + "_CLCT.nc")
nwp = xr.open_mfdataset(nwp_file)
x_1 = nwp.x_1.values
y_1 = nwp.y_1.values
errors = {"x_1": [], "y_1": [], "init_time": [], "time": [], "CRPS": []}
for h in range(121):
    timestring = (init_time + pd.Timedelta(hours=h)).strftime('%Y%m%d%H%M%S')
    print("Doing hour", h, "of 120")
    labels_df = xr.open_mfdataset(os.path.join(
        labels_dir, "meteosat.CFC.H_ch05.latitude_longitude_" + timestring + ".nc")).to_dataframe()
    time = pd.Timestamp(timestring)
    for x, y in itertools.product(x_1, y_1):
        tic = t.clock()
        predictions = np.array([nwp.CLCT[h].to_dataframe().loc[(i, y, x), "CLCT"] for i in range(nwp.epsd_1.size)])  # bottleneck
        print(t.clock() - t, "seconds")
        lon_lat = lookup_dict[(x, y)]
        observation = labels_df.loc[(lon_lat[1], lon_lat[0], time), "CFC"]
        error = ps.crps_ensemble(observation, predictions)
        errors["x_1"].append(x)
        errors["y_1"].append(y)
        errors["init_time"].append(init_time)
        errors["time"].append(time)
        errors["CRPS"].append(error)
df = pd.DataFrame(errors).set_index(["x_1", "y_1", "init_time", "time"])
df.to_pickle(os.path.join(save_dir, "NWP_CRPS_" + init_time_str.replace(' ', '') + ".pkl"))