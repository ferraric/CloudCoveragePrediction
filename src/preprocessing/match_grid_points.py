import xarray as xr
import numpy as np
import pickle

# change the following two lines, depending on your machine
labels = xr.open_mfdataset("play_data/meteosat.CFC.H_ch05.latitude_longitude_20180221*.nc")
predictions = xr.open_mfdataset("play_data/cosmo-e_2018022100_CLCT.nc")

lat = labels["lat"].values
lon = labels["lon"].values
lon_lat_grid = np.transpose([np.tile(lon, len(lat)), np.repeat(lat, len(lon))])
lon_lat_grid = [tuple(map(lambda a: round(a, 3), y)) for y in lon_lat_grid]  # 5500 pairs
prediction_coordinates = predictions[["lon_1", "lat_1"]].to_dataframe().values  # shape (23876, 2)
prediction_coordinates = np.unique(prediction_coordinates, axis=1)
lookup_table = {tuple(x): lon_lat_grid[np.argmin([np.linalg.norm(x - c, 2) for c in lon_lat_grid])]
                for x in prediction_coordinates}
f = open("lookup_table.pkl", "wb")
pickle.dump(lookup_table, f)
f.close()
