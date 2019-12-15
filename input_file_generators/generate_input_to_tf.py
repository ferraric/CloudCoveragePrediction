import pandas as pd
import numpy as np
column = "labels"
filepath = "/mnt/ds3lab-scratch/yidai/nwp_subsampled_2x2_5_day_stride_ensemble_mean_var/with_" + column 
init_time_00 = pd.read_pickle(filepath + "_it00.pkl")
init_time_12 = pd.read_pickle(filepath + "_it12.pkl")
# pick an arbitrary date to get dictionary of all possible lat lon pairs
#subset = init_time_12[init_time_12["time"] == "2014-02-02 12:00:00"]
#dict = {}
#for i in range(0, subset.shape[0]):
#    dict[(subset.iloc[i, 2], subset.iloc[i, 3])] = i
new_df_00 = pd.DataFrame()
new_df_12 = pd.DataFrame()
#new_df_00["CLCT_mean"] = init_time_00["CLCT_mean"]
#new_df_12["CLCT_mean"] = init_time_12["CLCT_mean"]
#new_df_00["CLCT_var"] = init_time_00["CLCT_var"]
#new_df_12["CLCT_var"] = init_time_12["CLCT_var"]
new_df_00[column] = init_time_00[column]
new_df_12[column] = init_time_12[column]
new_df_00["time"] = init_time_00["time"]
new_df_12["time"] = init_time_12["time"]
new_df_00["real_init_time"] = init_time_00["init_time"]
new_df_12["real_init_time"] = init_time_12["init_time"]
#new_df_00["lat_lon_id"] = [dict[(init_time_00.iloc[i, 3], init_time_00.iloc[i, 2])] for i in range(0, init_time_00.shape[0])]
#new_df_12["lat_lon_id"] = [dict[(init_time_12.iloc[i, 2], init_time_12.iloc[i, 3])] for i in range(0, init_time_12.shape[0])]
#new_df_00["hour"] = [x.hour for x in init_time_00["time"]]
#new_df_00["year"] = [x.year for x in init_time_00["time"]]
#new_df_00["month"] = [x.month for x in init_time_00["time"]]
#new_df_00["lead_time"] = init_time_00["lead_time"]
#new_df_00["init_time"] = [0 for i in range(0,init_time_00.shape[0])]
#new_df_12["hour"] = [x.hour for x in init_time_12["time"]]
#new_df_12["year"] = [x.year for x in init_time_12["time"]]
#new_df_12["month"] = [x.month for x in init_time_12["time"]]
#new_df_12["lead_time"] = init_time_12["lead_time"]
#new_df_12["init_time"] = [1 for i in range(0,init_time_12.shape[0])]
new_df_00 = new_df_00[np.isfinite(new_df_00[column])]
new_df_12 = new_df_12[np.isfinite(new_df_12[column])]
full_data = pd.concat([new_df_00,new_df_12],axis=0)
full_data.to_pickle("/mnt/ds3lab-scratch/yidai/input_to_tf_" + column + "_init_time.pkl")

