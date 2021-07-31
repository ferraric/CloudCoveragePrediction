import numpy as np
from typing import Tuple

lon_north_pole = -170
lat_north_pole = 43
lon_north_pole_ = np.radians(lon_north_pole)
lat_north_pole_ = np.radians(lat_north_pole)


def transform_to_x_y(lon: float, lat: float) -> Tuple[float, float]:
    """
    Transforms lon, lat coordinates in the standard coordinate system
        into x, y coordinates in the rotated coordinate system.
        Warning: the output is rounded, so precision is lost

    Args:
        lon:     Longitude in degrees in the standard coordinate system
                    of the point to be transformed
        lat:     Latitude in degrees in the standard coordinate system
                    of the point to be transformed


    Returns:
        x:      Longitude in degrees in the rotated coordinate system

        y:      Latitude in degrees in the rotated coordinate system

    """
    lon_ = np.radians(lon)
    lat_ = np.radians(lat)

    x_ = np.arctan(
        np.cos(lat_) * np.sin(lon_ - lon_north_pole_) /
        (np.cos(lat_) * np.sin(lat_north_pole_) *
         np.cos(lon_ - lon_north_pole_) -
         np.sin(lat_) * np.cos(lat_north_pole_)))
    y_ = np.arcsin(
        np.sin(lat_) * np.sin(lat_north_pole_) + np.cos(lat_) *
        np.cos(lat_north_pole_) * np.cos(lon_ - lon_north_pole_))
    return np.round(np.degrees(x_), 2), np.round(np.degrees(y_), 2)


def transform_to_lon_lat(x: float, y: float) -> Tuple[float, float]:
    """
    Transforms x, y coordinates in the rotated coordinate system
        into lon, lat coordinates in the standard coordinate system

    Args:
        x:     Longitude in degrees in the rotated coordinate system
                   of the point to be transformed

        y:     Latitude in degrees in the rotated coordinate system
                   of the point to be transformed

    Returns:
        lon:      Longitude in degrees in the standard coordinate system

        lat:      Latitude in degrees in the standard coordinate system

    """
    x_ = np.radians(x)
    y_ = np.radians(y)

    lon_ = np.arctan(
        np.cos(y_) * np.sin(x_) /
        (np.sin(lat_north_pole_) * np.cos(y_) * np.cos(x_) -
         np.sin(y_) * np.cos(lat_north_pole_))) + lon_north_pole_
    lat_ = np.arcsin(
        np.sin(y_) * np.sin(lat_north_pole_) +
        np.cos(y_) * np.cos(x_) * np.cos(lat_north_pole_))
    return np.degrees(lon_) + 180, np.degrees(lat_)
