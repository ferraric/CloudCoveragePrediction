import pickle
import xarray as xr
import pandas as pd
import properscoring as ps
import os
import numpy as np
from datetime import date, timedelta
import datetime, calendar


class ClimateBase:
    def __init__(self, year, month, day, hour, window):
        self.year = year
        self.month = month
        self.window = window
        self.day = day
        self.hour = hour

    def initialize_window(self):
        if (self.window == 1):
            self.window_dict = self.get_window1()
        elif (self.window == 2):
            self.window_dict = self.get_window2()
        else:
            self.window_dict = self.get_window3()

    def get_window1(self):
        window1 = {}
        window1["12"] = [12, 1, 2]  #winter
        window1["1"] = [12, 1, 2]
        window1["2"] = [12, 1, 2]
        window1["3"] = [3, 4, 5]  #Spring
        window1["4"] = [3, 4, 5]
        window1["5"] = [3, 4, 5]
        window1["6"] = [6, 7, 8]  #Summer
        window1["7"] = [6, 7, 8]
        window1["8"] = [6, 7, 8]
        window1["9"] = [9, 10, 11]  #Autumn
        window1["10"] = [9, 10, 11]
        window1["11"] = [9, 10, 11]
        return window1

    def get_window2(self):
        window2 = {}
        window2["12"] = [11, 12, 1]
        window2["1"] = [12, 1, 2]
        window2["2"] = [1, 2, 3]
        window2["3"] = [2, 3, 4]
        window2["4"] = [3, 4, 5]
        window2["5"] = [4, 5, 6]
        window2["6"] = [5, 6, 7]
        window2["7"] = [6, 7, 8]
        window2["8"] = [7, 8, 9]
        window2["9"] = [8, 9, 10]
        window2["10"] = [9, 10, 11]
        window2["11"] = [10, 11, 12]
        return window2

    def get_window3(self):
        window3 = {}
        window3["12"] = [10, 11, 12]
        window3["1"] = [11, 12, 1]
        window3["2"] = [12, 1, 2]
        window3["3"] = [1, 2, 3]
        window3["4"] = [2, 3, 4]
        window3["5"] = [3, 4, 5]
        window3["6"] = [4, 5, 6]
        window3["7"] = [5, 6, 7]
        window3["8"] = [6, 7, 8]
        window3["9"] = [7, 8, 9]
        window3["10"] = [8, 9, 10]
        window3["11"] = [9, 10, 11]
        return window3

    def return_quantiles(self, input_arr, max_quantiles):
        quantiles = []
        for i in range(1, max_quantiles + 1):
            quantiles.append(((i - 0.5) / max_quantiles))
        return np.quantile(input_arr, quantiles, axis=0)

    def add_days(self, n):
        d = datetime.datetime(self.year, self.month, self.day)
        DD = datetime.timedelta(days=n)
        new_date = d + DD
        return (new_date.year, new_date.month, new_date.day)

    def subtract_days(self, n):
        d = datetime.datetime(self.year, self.month, self.day)
        DD = datetime.timedelta(days=n)
        new_date = d - DD
        return (new_date.year, new_date.month, new_date.day)

    def get_all_days_between_dates(
            self, date_tuple1,
            date_tuple2):  #get list of all days between two given dates
        date_set = []
        sdate = date(date_tuple1[0], date_tuple1[1],
                     date_tuple1[2])  # start date
        edate = date(date_tuple2[0], date_tuple2[1],
                     date_tuple2[2])  # end date
        delta = edate - sdate  # as timedelta
        for i in range(delta.days + 1):
            day = sdate + timedelta(days=i)
            date_set.append((day.year, day.month, day.day))
        return date_set

    def get_all_days_between_months(
            self, date_tuple1,
            date_tuple2):  # month version of the earlier function...redundant?
        date_set = []
        sdate = date(date_tuple1[0], date_tuple1[1], 1)  # start date
        _, num_days = calendar.monthrange(date_tuple2[0], date_tuple2[1])
        edate = date(date_tuple2[0], date_tuple2[1], num_days)  # end date
        delta = edate - sdate  # as timedelta
        for i in range(delta.days + 1):
            day = sdate + timedelta(days=i)
            date_set.append((day.year, day.month, day.day))
        return date_set

    def startwith_12thmonth(self, start_month, mid_month,
                            end_month):  # handle the 12th month and year end
        all_days = []
        all_days_2014_p1 = self.get_all_days_between_months(
            (2014, int(start_month)), (2014, int(start_month)))
        all_days_2014_p2 = self.get_all_days_between_months(
            (2014, int(mid_month)), (2014, int(end_month)))
        all_days_2015_p1 = self.get_all_days_between_months(
            (2015, int(start_month)), (2015, int(start_month)))
        all_days_2015_p2 = self.get_all_days_between_months(
            (2015, int(mid_month)), (2015, int(end_month)))
        all_days_2016_p1 = self.get_all_days_between_months(
            (2016, int(start_month)), (2016, int(start_month)))
        all_days_2016_p2 = self.get_all_days_between_months(
            (2016, int(mid_month)), (2016, int(end_month)))
        all_days_2017_p1 = self.get_all_days_between_months(
            (2017, int(start_month)), (2017, int(start_month)))
        all_days_2017_p2 = self.get_all_days_between_months(
            (2017, int(mid_month)), (2017, int(end_month)))
        all_days_2018_p1 = self.get_all_days_between_months(
            (2018, int(start_month)), (2018, int(start_month)))
        all_days_2018_p2 = self.get_all_days_between_months(
            (2018, int(mid_month)), (2018, int(end_month)))
        all_days.extend(all_days_2014_p1 + all_days_2014_p2 +
                        all_days_2015_p1 + all_days_2015_p2 +
                        all_days_2016_p1 + all_days_2016_p2 +
                        all_days_2017_p1 + all_days_2017_p2 +
                        all_days_2018_p1 + all_days_2018_p2)
        #print(all_days)
        return all_days

    def startwith_11thmonth(self, start_month, mid_month,
                            end_month):  #handle the 11th month
        all_days = []
        all_days_2014_p1 = self.get_all_days_between_months(
            (2014, int(start_month)), (2014, int(mid_month)))
        all_days_2014_p2 = self.get_all_days_between_months(
            (2014, int(end_month)), (2014, int(end_month)))
        all_days_2015_p1 = self.get_all_days_between_months(
            (2015, int(start_month)), (2015, int(mid_month)))
        all_days_2015_p2 = self.get_all_days_between_months(
            (2015, int(end_month)), (2015, int(end_month)))
        all_days_2016_p1 = self.get_all_days_between_months(
            (2016, int(start_month)), (2016, int(mid_month)))
        all_days_2016_p2 = self.get_all_days_between_months(
            (2016, int(end_month)), (2016, int(end_month)))
        all_days_2017_p1 = self.get_all_days_between_months(
            (2017, int(start_month)), (2017, int(mid_month)))
        all_days_2017_p2 = self.get_all_days_between_months(
            (2017, int(end_month)), (2017, int(end_month)))
        all_days_2018_p1 = self.get_all_days_between_months(
            (2018, int(start_month)), (2018, int(mid_month)))
        all_days_2018_p2 = self.get_all_days_between_months(
            (2018, int(end_month)), (2018, int(end_month)))
        all_days.extend(all_days_2014_p1 + all_days_2014_p2 +
                        all_days_2015_p1 + all_days_2015_p2 +
                        all_days_2016_p1 + all_days_2016_p2 +
                        all_days_2017_p1 + all_days_2017_p2 +
                        all_days_2018_p1 + all_days_2018_p2)
        #print(all_days)
        return all_days

    def startwith_normal(self, start_month, end_month):
        all_days = []
        all_days_2014 = self.get_all_days_between_months(
            (2014, int(start_month)), (2014, int(end_month)))
        all_days_2015 = self.get_all_days_between_months(
            (2015, int(start_month)), (2015, int(end_month)))
        all_days_2016 = self.get_all_days_between_months(
            (2016, int(start_month)), (2016, int(end_month)))
        all_days_2017 = self.get_all_days_between_months(
            (2017, int(start_month)), (2017, int(end_month)))
        all_days_2018 = self.get_all_days_between_months(
            (2018, int(start_month)), (2018, int(end_month)))
        all_days.extend(all_days_2014 + all_days_2015 + all_days_2016 +
                        all_days_2017 + all_days_2018)
        return all_days

    def get_date_path(self, base, correct_day, hourstr):
        if correct_day[1] < 10 and correct_day[2] < 10:  # file name construct
            dpath = base + str(correct_day[0]) + "0" + str(
                correct_day[1]) + "0" + str(correct_day[2]) + hourstr
        elif correct_day[1] < 10 and correct_day[2] >= 10:
            dpath = base + str(correct_day[0]) + "0" + str(
                correct_day[1]) + str(correct_day[2]) + hourstr
        elif correct_day[1] >= 10 and correct_day[2] < 10:
            dpath = base + str(correct_day[0]) + str(
                correct_day[1]) + "0" + str(correct_day[2]) + hourstr
        else:
            dpath = base + str(correct_day[0]) + str(correct_day[1]) + str(
                correct_day[2]) + hourstr
        return dpath
