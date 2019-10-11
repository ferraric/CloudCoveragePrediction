import numpy as np
import xarray as xr
import coordinate_transformer

labels = xr.open_mfdataset("../../local_data/labels/meteosat.CFC.H_ch05.latitude_longitude_201812*.nc",
                           combine='by_coords').CFC

max_lat = np.max(labels.lat.values)
min_lat = np.min(labels.lat.values)
max_lon = np.max(labels.lon.values)
min_lon = np.min(labels.lon.values)
print(max_lon, max_lat)
print(min_lon, max_lat)
print(max_lon, min_lat)
print(min_lon, min_lat)
print(coordinate_transformer.transform_to_x_y(max_lon, max_lat))
print(coordinate_transformer.transform_to_x_y(min_lon, max_lat))
print(coordinate_transformer.transform_to_x_y(max_lon, min_lat))
print(coordinate_transformer.transform_to_x_y(min_lon, min_lat))
