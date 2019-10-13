import pandas as pd
import os

folder = os.path.join("~", "evaluation")
dates = pd.date_range("20140101 00", "20181226 12", freq="12H")
file_names = ["persistence_CRPS_" + date.strftime("%Y%m%d%H") + ".pkl" for date in dates]
no_consecutive = 0
for name in file_names:
    try:     
        pd.read_pickle(os.path.join(folder, name))
        if no_consecutive > 0:
            print(no_consecutive)
            no_consecutive = 0
    except FileNotFoundError:
        print(name)
        no_consecutive += 1
