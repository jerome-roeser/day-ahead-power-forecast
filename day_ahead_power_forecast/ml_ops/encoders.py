import numpy as np
import pandas as pd


def time_features_encoder(X: pd.DataFrame, time_column_name) -> pd.DataFrame:
    """
    Convert local time into cyclic features to feed significant signal
    in ML / DL algorithm

    Arguments:
        X (pd.DataFrame): datafrane to transform

    Returns:
        X (pd.DataFrame): DataFrame with 4 addtional features per time column
    """
    X = X.copy()

    local_time = pd.to_datetime(X.pop(time_column_name), utc=True)
    timestamp_s = local_time.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    year = (365.2425) * day
    X["day_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    X["day_cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    X["year_sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    X["year_cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    return X.iloc[:, -4:]


def wind_features_encoder(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert local time into cyclic features to feed significant signal
    in ML / DL algorithm

    Args:
        X: datafrane to transform

    Returns:
        DataFrame with 4 addtional features per time column
    """
    X = X.copy()

    # Process wind fratures
    wind_speed = X.pop("wind_speed")

    # Convert to radians.
    wind_rad = X.pop("wind_deg") * np.pi / 180

    # Calculate the wind x and y components
    X["Wx"] = wind_speed * np.cos(wind_rad)
    X["Wy"] = wind_speed * np.sin(wind_rad)

    # Standardize the components
    X["Wx"] = (X["Wx"] - X["Wx"].mean()) / X["Wx"].std()
    X["Wy"] = (X["Wy"] - X["Wy"].mean()) / X["Wy"].std()

    return X.iloc[:, -2:]
