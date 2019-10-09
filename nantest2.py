import numpy as np
import pandas as pd
import xarray as xr
import pickle
top = 1
fourteen = []
nanfourteen = []
fifteen = []
nanfifteen = []
sixteen = []
nansixteen = []
seventeen = []
nanseventeen = []
eighteen = []
naneighteen = []
set1 = {1, 2, 3, 4, 5, 6, 7, 8, 9}
years = ['2014', '2015', '2016', '2017', '2018']
months = [
    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'
]
times = [
    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '00'
]
base = "/mnt/ds3lab-scratch/bhendj/grids/CM-SAF/MeteosatCFC/"
#March
for x in years:
    for y in months:
        for z in times:
            if (y == "03" or y == "05" or y == "01" or y == "07" or y == "08"
                    or y == "10" or y == "12"):
                for i in range(1, 32):
                    if x == "2014":
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2014" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2014" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nanfourteen.append(dpath)
                        fourteen.append(df["CFC"].values)
                    if x == "2015":
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2015" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2015" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nanfifteen.append(dpath)
                        fifteen.append(df["CFC"].values)
                    if x == "2016":
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2016" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2016" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nansixteen.append(dpath)
                        sixteen.append(df["CFC"].values)
                    if x == "2017":
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2017" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2017" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nanseventeen.append(dpath)
                        seventeen.append(df["CFC"].values)
                    if x == "2018":
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2018" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2018" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            naneighteen.append(dpath)
                        eighteen.append(df["CFC"].values)
            elif (y == "04" or y == "06" or y == "09" or y == "11"):
                for i in range(1, 31):
                    if x == "2014":
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2014" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2014" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nanfourteen.append(dpath)
                        fourteen.append(df["CFC"].values)
                    if x == "2015":
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2015" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2015" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nanfifteen.append(dpath)
                        fifteen.append(df["CFC"].values)
                    if x == "2016":
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2016" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2016" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nansixteen.append(dpath)
                        sixteen.append(df["CFC"].values)
                    if x == "2017":
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2017" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2017" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nanseventeen.append(dpath)
                        seventeen.append(df["CFC"].values)
                    if x == "2018":
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2018" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2018" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            naneighteen.append(dpath)
                        eighteen.append(df["CFC"].values)
            else:
                for i in range(1, 30):
                    if (x == "2014" and i != 29):
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2014" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2014" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nanfourteen.append(dpath)
                        fourteen.append(df["CFC"].values)
                    if (x == "2015" and i != 29):
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2015" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2015" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nanfifteen.append(dpath)
                        fifteen.append(df["CFC"].values)
                    if x == "2016":
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2016" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2016" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nansixteen.append(dpath)
                        sixteen.append(df["CFC"].values)
                    if (x == "2017" and i != 29):
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2017" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2017" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            nanseventeen.append(dpath)
                        seventeen.append(df["CFC"].values)
                    if (x == "2018" and i != 29):
                        if i in set1:
                            dpath = base + "meteosat.CFC.H_ch05.latitude_longitude_2018" + y + "0" + str(
                                i) + z + "0000.nc"
                        else:
                            dpath = base + "/meteosat.CFC.H_ch05.latitude_longitude_2018" + y + str(
                                i) + z + "0000.nc"
                        dat = xr.open_dataset(dpath)
                        df = dat.to_dataframe()
                        if sum(np.isnan(df["CFC"].values)) != 0:
                            print(df.head())
                            naneighteen.append(dpath)
                        eighteen.append(df["CFC"].values)

with open('nan14.pkl', 'wb') as f:
    pickle.dump(nanfourteen, f)
with open('nan15.pkl', 'wb') as f:
    pickle.dump(nanfifteen, f)
with open('nan16.pkl', 'wb') as f:
    pickle.dump(nansixteen, f)
with open('nan17.pkl', 'wb') as f:
    pickle.dump(nanseventeen, f)
with open('nan18.pkl', 'wb') as f:
    pickle.dump(naneighteen, f)

print("Statistics")
print("Nans in 14", len(nanfourteen))
print("Nans in 15", len(nanfifteen))
print("Nans in 16", len(nansixteen))
print("Nans in 17", len(nanseventeen))
print("Nans in 18", len(naneighteen))
