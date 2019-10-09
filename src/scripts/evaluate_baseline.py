import xarray as xr
import pandas as pd
import properscoring as ps
import os

# change the following lines, depending on your machine and time range
save_dir = os.path.join("~", "evaluation")
time_range = "2018"
labels_file = "/mnt/ds3lab-scratch/bhendj/grids/CM-SAF/MeteosatCFC/meteosat.CFC.H_ch05.latitude_longitude_" \
              + time_range + "*.nc"

labels = xr.open_mfdataset(labels_file)
labels_df = labels.to_dataframe()
keys = labels_df.index.values
errors = {"lat": [], "lon": [], "time": [], "CRPS": []}
for lat, lon, time in keys:
    try:
        prediction = labels_df.loc[(lat, lon, time - pd.Timedelta('01:00:00')), "CFC"]
        observation = labels_df.loc[(lat, lon, time), "CFC"]
        error = ps.crps_ensemble(observation, prediction)  # reduces to abs error in this case
    except KeyError:
        error = 0
    errors["lat"].append(lat)
    errors["lon"].append(lon)
    errors["time"].append(time)
    errors["CRPS"].append(error)
df = pd.DataFrame(errors).set_index(["lat", "lon", "time"])
df.to_pickle(os.path.join(save_dir, "persistence_CRPS_" + time_range + ".pkl"))
