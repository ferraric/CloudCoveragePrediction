#!/usr/bin/python
from pudb import set_trace
import random
import numpy as np
from datetime import datetime, timedelta

import xarray as xr
import pymchdwh as dwh
import pyproj
from scipy import spatial


parameter_codes = {
    'wind_speed': 'fkl010z0',
    'wind_direction': 'dkl010z0'
}


class StationSelector(object):
    """
    Selects the indices of gridcells that are closest to the stations provided.
    """
    def __init__(self, dataset):
        try:
            lon = dataset['COSMO-E.lon'].values
            lat = dataset['COSMO-E.lat'].values
        except:
            lon = dataset['lon'].values
            lat = dataset['lat'].values
        self.model_proj = pyproj.Proj(init='EPSG:4326')
        self.target_proj = pyproj.Proj(init='EPSG:2056')
        self.shape = lon.shape

        # Reshape coordinates into 2 column matrix
        coords = np.column_stack((lon.ravel(), lat.ravel()))

        # Construct the KD-tree
        self.tree = spatial.cKDTree(self._transform_coordinates(coords))

    def _transform_coordinates(self, coords):
        """
        Transform coordinates to swiss coordinate system.
        """
        transformed = pyproj.transform(
            self.model_proj,
            self.target_proj,
            coords[:, 0],
            coords[:, 1]
        )

        # Need to transpose because x and y are provided as row vectors
        return np.asarray(transformed).T

    def query(self, stations):
        """
        Get the nearest neighbour grid cell of a station

        Args:
            points:     list of station names to be queried.

        Returns:
            x:          DataArray, containing x indices of nearest cells of
                        queried points.

            y:          DataArray, containing y indices of nearest cells of
                        queried points.

        """
        # Get coordinates of stations
        coords= get_coordinates(stations)

        # Query for nearest neighbour
        points = np.array(coords)
        _, index = self.tree.query(self._transform_coordinates(points))

        # Regrid to 2D grid
        index = np.unravel_index(index, self.shape)

        x = xr.DataArray(
            index[1],
            dims='station',
            coords={'station': stations}
        )
        y = xr.DataArray(index[0], dims='station')
        return x, y


class Standardizer(object):
    """
    Standardizes data according to a xarray dataset.
    Based on the StandardScaler class of scikit learn
    """
    def fit(self, dataset):
        self.mean = dataset.mean()
        self.std = dataset.std()

    def transform(self, dataset):
        return (dataset - self.mean)/self.std


def transform_dataset(dataset, stations, reftimes, leadtimes, inputs, targets):
    """
    Transforms all features and targets according to specifications.
    """
    new_set = xr.Dataset()
    for feature in inputs:
        name = "data." + feature
        if feature == 'dayofyear':
            array = dataset.reftime.dt.dayofyear
        elif feature == 'hourofday':
            array = dataset.reftime.dt.hour
        elif feature == 'lon':
            array = dataset['lon']
        elif feature == 'lat':
            array = dataset['lat']
        elif feature == 'leadtime':
            array = dataset['leadtime']
        elif feature == 'reftime':
            array = dataset.reftime

        # Broadcast together with member dimensions
        new_set[name] = array.broadcast_like(dataset.member)

    for target in targets:
        name = "data." + target
        new_set[name] = dataset[target]

    return new_set


def get_wind_observations(stations, reftimes, leadtimes, name):
    da = None
    for station in stations:
        station_data = None
        for reftime in reftimes:
            time_axis = dwh.TimeAxis(
                [reftime + timedelta(hours=lead) for lead in leadtimes]
            )
            wind_speed = dwh.surface(
                parameter_codes['wind_speed'],
                station,
                time_axis
            )
            wind_direction = dwh.surface(
                parameter_codes['wind_direction'],
                station,
                time_axis
            )
            # Rename columns to allow multiplication
            wind_speed = wind_speed.rename(
                {parameter_codes['wind_speed']: 'data'},
                axis=1
            )
            wind_direction = wind_direction.rename(
                {parameter_codes['wind_direction']: 'data'},
                axis=1
            )

            if 'northward' in name:
                factor = wind_direction.apply(lambda x: x*np.pi/180).apply(np.cos)
            else:
                factor = wind_direction.apply(lambda x: x*np.pi/180).apply(np.sin)
            wind = wind_speed.multiply(factor)
            observations = xr.DataArray(
                name='data.' + name,
                data=wind,
                coords={'reftime': [reftime]},
                dims=['leadtime', 'reftime']
            )

            # Concatenate reftimes
            if station_data is None:
                station_data = observations
            else:
                station_data = xr.concat(
                    [station_data, observations],
                    dim='reftime'
                )

        # Add station dimension
        station_data = station_data.expand_dims({'station': [station]})

        # Concatenate stations
        if da is None:
            da = station_data
        else:
            da = xr.concat(
                [da, station_data],
                dim='station'
            )

    return da


def reshape_rename_dataset(dataset):
    try:
        # Drop variables that are not needed
        dataset = dataset.drop_dims(['validtime'])

        # Rename reftime variable and turn into dimension
        dataset = dataset.rename({'COSMO-E.data.reftime': 'reftime'})
        dataset = dataset.swap_dims({'data': 'reftime'})

        # Rename targets
        for target in targets:
            dataset = dataset.rename({f"COSMO-E.data.{target}": f"{target}"})
    except:
        pass

    return dataset


def clean_station_list(stations):
    # Removes all stations that are not in the dwh database
    for station in stations:
        info = dwh.poi_info(station)
        if info is None:
            stations.remove(station)

    return stations


def preprocess_dataset(
    dataset,
    inputs,
    targets,
    stations,
    reftimes,
    leadtimes,
    reftimes_train,
    stations_train
):
    """
    Modify dataset for use in ML
    """
    # Select all the grid cells closest to the stations
    selector = StationSelector(dataset)
    x_indices, y_indices = selector.query(stations)
    dataset = dataset.sel(x=x_indices, y=y_indices)

    dataset = reshape_rename_dataset(dataset)

    # Create new features and targets
    dataset = transform_dataset(
        dataset,
        stations,
        reftimes,
        leadtimes,
        inputs,
        targets
    )

    # Standardize according to training split
    train_dataset = dataset.sel(
        reftime=reftimes_train,
        station=stations_train
    )
    standardizer = Standardizer(train_dataset)

    return dataset, standardizer


def get_coordinates(stations):
    # Extract coordinates of stations from dwh
    set_trace()
    coordinates = []
    for station in stations:
        print(station)
        info = dwh.poi_info(station)
        coordinates.append((info['longitude'], info['latitude']))

    return coordinates

