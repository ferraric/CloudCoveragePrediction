from ClimateBase import ClimateBase 
import numpy as np
import xarray as xr
import os
import pickle
import properscoring as ps
from datetime import datetime,date, timedelta
import calendar

def main():
    max_quantiles = 21
    years = ['2014','2015','2016','2017','2018']#initialize all years
    prohibited_window_size=10  #number of days before and after current date to be ignored
    months = ['1', '2', '3', '4', '5','6','7','8','9','10','11','12']#initialize all months
    window=1
    hours = ['00']#'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23'];#not written in usual form for regexp construction
    data_path="/mnt/ds3lab-scratch/bhendj/grids/CM-SAF/MeteosatCFC/meteosat.CFC.H_ch05.latitude_longitude_"#define path of data
    results_path= "results_test/"#define path to save results
    for y in years:
        for m in months:
            _,num_days_in_month=calendar.monthrange(int(y),int(m))# get number of days in current month of current year
            for day in range(1,num_days_in_month+1):
                for hour in hours:
                    cb_obj=ClimateBase(int(y),int(m),int(day),int(hour),window)
                    hourstr = hour + "0000.nc" #construct file path for hour
                    ten_days_earlier=cb_obj.add_days(prohibited_window_size)
                    ten_days_after=cb_obj.subtract_days(prohibited_window_size)
                    prohibited_list=cb_obj.get_all_days_between_dates(ten_days_earlier, ten_days_after)
                    cb_obj.initialize_window()
                    months_to_avg =cb_obj.window_dict[m]
                    #print(months_to_avg)
                    start_month = months_to_avg[0]
                    mid_month = months_to_avg[1]
                    end_month = months_to_avg[2]
                    if (start_month ==12):
                        all_days=cb_obj.startwith_12thmonth(start_month,mid_month,end_month)
                    elif (start_month ==11):
                        all_days=cb_obj.startwith_11thmonth(start_month,mid_month,end_month)
                    else:
                        all_days=cb_obj.startwith_normal(start_month,end_month)
                    #print(all_days)
                    filtered_days = list(set(all_days) - set(prohibited_list))# filter out the dates 10 days before and after
                    #print(filtered_days)
                    CFC_values=[]
                    for correct_day in filtered_days:#mainly to construct file name and read file
                        dpath=cb_obj.get_date_path(data_path,correct_day,hourstr)
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        CFC=df["CFC"].values
                        #print(CFC)
                        if sum(np.isnan(CFC)) == 0: #check for nans
                            CFC_values.append(CFC)
                            #print(dpath)
                            lat = dat["lat"].values
                            lon = dat["lon"].values
                    out= np.array(CFC_values)
                    actpath =cb_obj.get_date_path(data_path,[int(y),int(m),int(day)],hourstr)
                    dat_act = xr.open_dataset(actpath)
                    #CFC_act=dat_act.CFC.values
                    df_act = dat_act.to_dataframe()
                    CFC_act=df_act["CFC"].values
                    if sum(np.isnan(CFC_act)) == 0:
                        actual = CFC_act
                        #print(actual.shape)
                        #print(out.shape)
                        #actual=np.reshape(actual,[np.shape(out)[1],1])
                        #print(np.shape(actual))
                        #print(np.shape(np.transpose(out)))
                        out =cb_obj.return_quantiles(out, max_quantiles)
                        CRPS = ps.crps_ensemble(actual, np.transpose(out))
                        result_dict = {}
                        result_dict["lat"] = lat
                        result_dict["lon"] = lon
                        result_dict["CRPS"] = CRPS
                        result_dict["FULL_OUTPUT"] = out
                        result_dict["time"] =datetime(int(y),int(m),int(day),int(hour))
                        print(result_dict)
                        path =results_path+ result_dict["time"].strftime('%Y%m%d%H') + ".pkl"#full result path
                        with open(path, 'wb') as f:
                            pickle.dump(result_dict, f)







if __name__ == '__main__':
    main()
