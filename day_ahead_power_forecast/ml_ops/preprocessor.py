import numpy as np
import pandas as pd
from colorama import Fore, Style
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from day_ahead_power_forecast.ml_ops.encoders import (
    time_features_encoder,
    wind_features_encoder,
)


def preprocess_PV_features(X: pd.DataFrame) -> np.ndarray:
    time_encoder = FunctionTransformer(
        time_features_encoder, kw_args={"time_column_name": "local_time"}
    )

    pv_transformer = ColumnTransformer(
        [
            ("Time Encoder", time_encoder, ["local_time"]),
            ("Passthrough", "passthrough", ["local_time", "electricity"]),
        ],
        remainder="drop",
    ).set_output(transform="pandas")

    print(Fore.BLUE + "\nPreprocessing PV features..." + Style.RESET_ALL)

    preprocess_pv = Pipeline([("Time Features", pv_transformer)])
    X_processed = preprocess_pv.fit_transform(X)

    X_processed.columns = [column.split("__")[1] for column in X_processed.columns]

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed


def preprocess_forecast_features(X: pd.DataFrame) -> np.ndarray:
    std_features = [
        "temperature",
        "dew_point",
        "pressure",
        "ground_pressure",
        "humidity",
    ]
    minmax_features = [
        "clouds",
        "rain",
        "snow",
        "ice",
        "fr_rain",
        "convective",
        "snow_depth",
        "accumulated",
        "hours",
        "rate",
        "probability",
    ]

    time_encoder = FunctionTransformer(
        time_features_encoder, kw_args={"time_column_name": "prediction_utc_time"}
    )
    wind_encoder = FunctionTransformer(wind_features_encoder)

    forecast_transformer = ColumnTransformer(
        [
            ("Time Encoder", time_encoder, ["prediction_utc_time"]),
            ("Wind Encoder", wind_encoder, ["wind_speed", "wind_deg"]),
            ("Std", StandardScaler(), std_features),
            ("MinMax", MinMaxScaler(), minmax_features),
            ("Passthrough", "passthrough", ["utc_time", "prediction_utc_time"]),
        ],
        remainder="passthrough",
    ).set_output(transform="pandas")

    print(Fore.BLUE + "\nPreprocessing Weather Forecast features..." + Style.RESET_ALL)

    X_processed = forecast_transformer.fit_transform(X)
    X_processed.columns = [column.split("__")[1] for column in X_processed.columns]

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed
