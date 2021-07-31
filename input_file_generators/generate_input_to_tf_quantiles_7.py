import pandas as pd
import numpy as np

column = "labels"
filepath = "/mnt/ds3lab-scratch/yidai/nwp_subsampled_2x2_5_day_stride_ensemble_7quantiles/2018_with_" + column
init_time_00 = pd.read_pickle(filepath + "_it00.pkl")
init_time_12 = pd.read_pickle(filepath + "_it12.pkl")
# pick an arbitrary date to get dictionary of all possible lat lon pairs
subset = init_time_12[init_time_12["time"] == "2018-01-02 12:00:00"]
print(subset.shape)
dict = {}
for i in range(0, subset.shape[0]):
    #print(subset["lat_1"])
    #print(subset["lon_1"])
    dict[(subset["lat_1"].iloc[i], subset["lon_1"].iloc[i])] = i
new_df_00 = pd.DataFrame()
new_df_12 = pd.DataFrame()
new_df_00["CLCT_quantile0"] = init_time_00["CLCT_quantile0"]
new_df_12["CLCT_quantile0"] = init_time_12["CLCT_quantile0"]
new_df_00["CLCT_quantile1"] = init_time_00["CLCT_quantile1"]
new_df_12["CLCT_quantile1"] = init_time_12["CLCT_quantile1"]
new_df_00["CLCT_quantile2"] = init_time_00["CLCT_quantile2"]
new_df_12["CLCT_quantile2"] = init_time_12["CLCT_quantile2"]
new_df_00["CLCT_quantile3"] = init_time_00["CLCT_quantile3"]
new_df_12["CLCT_quantile3"] = init_time_12["CLCT_quantile3"]
new_df_00["CLCT_quantile4"] = init_time_00["CLCT_quantile4"]
new_df_12["CLCT_quantile4"] = init_time_12["CLCT_quantile4"]
new_df_00["CLCT_quantile5"] = init_time_00["CLCT_quantile5"]
new_df_12["CLCT_quantile5"] = init_time_12["CLCT_quantile5"]
new_df_00["CLCT_quantile6"] = init_time_00["CLCT_quantile6"]
new_df_12["CLCT_quantile6"] = init_time_12["CLCT_quantile6"]
new_df_00["labels"] = init_time_00["labels"]
new_df_12["labels"] = init_time_12["labels"]
new_df_00["lat_lon_id"] = [
    dict[(init_time_00["lat_1"][i], init_time_00["lon_1"][i])]
    for i in range(0, init_time_00.shape[0])
]
new_df_12["lat_lon_id"] = [
    dict[(init_time_12["lat_1"][i], init_time_12["lon_1"][i])]
    for i in range(0, init_time_12.shape[0])
]
new_df_00["hour"] = [x.hour for x in init_time_00["time"]]
new_df_00["init_time_day"] = [x.day for x in init_time_00["init_time"]]
new_df_00["year"] = [x.year for x in init_time_00["time"]]
new_df_00["month"] = [x.month for x in init_time_00["time"]]
new_df_00["lead_time"] = init_time_00["lead_time"]
new_df_00["init_time"] = [0 for i in range(0, init_time_00.shape[0])]
new_df_12["hour"] = [x.hour for x in init_time_12["time"]]
new_df_12["init_time_day"] = [x.day for x in init_time_12["init_time"]]
new_df_12["year"] = [x.year for x in init_time_12["time"]]
new_df_12["month"] = [x.month for x in init_time_12["time"]]
new_df_12["lead_time"] = init_time_12["lead_time"]
new_df_12["init_time"] = [1 for i in range(0, init_time_12.shape[0])]
new_df_00 = new_df_00[np.isfinite(new_df_00["labels"])]
new_df_12 = new_df_12[np.isfinite(new_df_12["labels"])]
full_data = pd.concat([new_df_00, new_df_12], axis=0)
full_data.to_pickle("2018_test_set" + column + ".pkl")
