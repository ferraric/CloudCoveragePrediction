import xarray as xr
import pandas as pd
import properscoring as ps
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# change the following lines, depending on your machine and time range
save_dir = os.path.join("~", "evaluation")
init_time_str = "20181103 00"
labels_dir = os.path.join("/mnt", "ds3lab-scratch", "bhendj", "grids", "CM-SAF", "MeteosatCFC")
no_of_init_times = 200

for i in range(no_of_init_times):
    init_time = pd.Timestamp(init_time_str)
    persisted_labels_file = os.path.join(labels_dir,
                                         "meteosat.CFC.H_ch05.latitude_longitude_" + init_time_str.replace(' ', '') + "*.nc")
    persisted_labels = xr.open_mfdataset(persisted_labels_file)
    assert persisted_labels.dims["time"] == 1
    persisted_labels_df = persisted_labels.to_dataframe().reset_index("time")
    timelist = [(init_time + pd.Timedelta(hours=h)).strftime('%Y%m%d%H%M%S') for h in range(121)]
    keys = persisted_labels_df.index.values
    errors = {"lat": [], "lon": [], "init_time": [], "time": [], "CRPS": []}
    for timestring in timelist:
        print("Doing hour", timestring[:-4], "[", timelist[0][:-4], "-", timelist[-1][:-4], "];", "init time", i + 1, "of", no_of_init_times)
        obs_labels_df = xr.open_mfdataset(
            os.path.join(labels_dir, "meteosat.CFC.H_ch05.latitude_longitude_" + timestring + ".nc")).to_dataframe()
        time = pd.Timestamp(timestring)
        for lat, lon in keys:
            prediction = persisted_labels_df.loc[(lat, lon), "CFC"]
            observation = obs_labels_df.loc[(lat, lon, time), "CFC"]
            error = ps.crps_ensemble(observation, prediction)  # reduces to abs error in this case
            errors["lat"].append(lat)
            errors["lon"].append(lon)
            errors["init_time"].append(init_time)
            errors["time"].append(time)
            errors["CRPS"].append(error)
    df = pd.DataFrame(errors).set_index(["lat", "lon", "init_time", "time"])
    df.to_pickle(os.path.join(save_dir, "persistence_CRPS_" + init_time_str.replace(' ', '') + ".pkl"))
    init_time_str = (init_time + pd.Timedelta(hours=12)).strftime('%Y%m%d %H')

