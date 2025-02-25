import math
import numpy as np
import pandas as pd


def time_features_encoder(X: pd.DataFrame, time_column_name) -> pd.DataFrame:
    """
    Convert local time into cyclic features to feed significant signal
    in ML / DL algorithm

    Args:
        X (pd.DataFrame): datafrane to transform

    Returns:
        DataFrame with 4 addtional features per time column
    """
    X = X.copy()

    local_time = pd.to_datetime(X.pop(time_column_name), utc= True)
    timestamp_s = local_time.map(pd.Timestamp.timestamp)


    day = 24*60*60
    year = (365.2425)*day
    X['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    X['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    X['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    X['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return X.iloc[:,-4:]

def wind_features_encoder(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert local time into cyclic features to feed significant signal
    in ML / DL algorithm

    Args:
    X: datafrane to transform

    Output:
    DataFrame with 4 addtional features per time column
    """
    X = X.copy()

    # Process wind fratures
    wind_speed = X.pop('wind_speed')

    # Convert to radians.
    wind_rad = X.pop('wind_deg')*np.pi / 180

    # Calculate the wind x and y components
    X['Wx'] = wind_speed*np.cos(wind_rad)
    X['Wy'] = wind_speed*np.sin(wind_rad)

    # Standardize the components
    X['Wx'] = (X['Wx'] - X['Wx'].mean())/X['Wx'].std()
    X['Wy'] = (X['Wy'] - X['Wy'].mean())/X['Wy'].std()

    return X.iloc[:,-2:]

def transform_time_features(X: pd.DataFrame) -> np.ndarray:
    assert isinstance(X, pd.DataFrame)

    timedelta = (X["pickup_datetime"] - pd.Timestamp('2009-01-01T00:00:00', tz='UTC')) / pd.Timedelta(1,'D')

    pickup_dt = X["pickup_datetime"].dt.tz_convert("America/New_York").dt
    dow = pickup_dt.weekday
    hour = pickup_dt.hour
    month = pickup_dt.month

    hour_sin = np.sin(2 * math.pi / 24 * hour)
    hour_cos = np.cos(2*math.pi / 24 * hour)

    return np.stack([hour_sin, hour_cos, dow, month, timedelta], axis=1)


def transform_lonlat_features(X: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(X, pd.DataFrame)
    lonlat_features = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

    def distances_vectorized(df: pd.DataFrame, start_lat: str, start_lon: str, end_lat: str, end_lon: str) -> dict:
        """
        Calculate the haversine and Manhattan distances between two
        points on the earth (specified in decimal degrees)
        Vectorized version for pandas df
        Computes distance in km
        """
        earth_radius = 6371

        lat_1_rad, lon_1_rad = np.radians(df[start_lat]), np.radians(df[start_lon])
        lat_2_rad, lon_2_rad = np.radians(df[end_lat]), np.radians(df[end_lon])

        dlon_rad = lon_2_rad - lon_1_rad
        dlat_rad = lat_2_rad - lat_1_rad

        manhattan_rad = np.abs(dlon_rad) + np.abs(dlat_rad)
        manhattan_km = manhattan_rad * earth_radius

        a = (np.sin(dlat_rad / 2.0)**2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon_rad / 2.0)**2)
        haversine_rad = 2 * np.arcsin(np.sqrt(a))
        haversine_km = haversine_rad * earth_radius

        return dict(
            haversine=haversine_km,
            manhattan=manhattan_km
        )

    result = pd.DataFrame(distances_vectorized(X, *lonlat_features))

    return result
