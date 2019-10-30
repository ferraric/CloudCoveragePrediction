import xarray as xr
import pickle

# arbitrary nwp file containing clct data
predictions = xr.open_mfdataset(
    "../local/local_data/NWP_output/cosmo-e_2018122500_CLCT.nc")
clct_predictions = predictions.CLCT
# index is now y, x and values are lon, lat
yx_to_ll_df = clct_predictions.to_dataframe().reset_index(["time", "epsd_1"])[['lon_1', 'lat_1']]
yx_to_ll_dict = dict(zip(yx_to_ll_df.index.values, yx_to_ll_df.values))

f = open("../shared_data/yx_to_ll_dict.pkl", "wb")
pickle.dump(yx_to_ll_dict, f)
f.close()
