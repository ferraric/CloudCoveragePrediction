import xarray as xr
import numpy as np
import pickle, os

def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y), 2)

def write_pkl(object, filepath):
    f = open(filepath, "wb")
    pickle.dump(object, f)
    f.close()


# change the following two lines, depending on your machine
labels = xr.open_mfdataset("../local_playground/meteosat.CFC.H_ch05.latitude_longitude_20180221*.nc")
predictions = xr.open_mfdataset("../local_playground/cosmo-e_2018022100_CLCT.nc")

lat = labels["lat"].values
lon = labels["lon"].values
lat_lon_grid = np.transpose([np.tile(lat, len(lon)), np.repeat(lon, len(lat))])  # 5500 pairs
prediction_df = predictions[["lon_1", "lat_1"]].to_dataframe()
prediction_coordinates = prediction_df.values[:, ::-1]  # shape (23876, 2)
prediction_coordinates = np.unique(prediction_coordinates, axis=1)
prediction_xy = prediction_df.index.values

data_path = os.path.join("..", "..", "shared_data")
lookup_table_ll_ll = {tuple(x): lat_lon_grid[np.argmin([distance(x, c) for c in lat_lon_grid])]
                      for x in prediction_coordinates}
write_pkl(lookup_table_ll_ll, os.path.join(data_path, "lookup_ll_ll.pkl"))

lookup_table_xy_ll = {tuple(x): lookup_table_ll_ll[tuple(prediction_df.loc[x, :])] for x in prediction_xy}
write_pkl(lookup_table_xy_ll, os.path.join(data_path, "lookup_xy_ll.pkl"))
