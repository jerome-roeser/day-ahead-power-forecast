from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from day_ahead_power_forecast.interface.main import pred
from day_ahead_power_forecast.ml_ops.data import (
    get_data_with_cache,
    get_stats_table,
    postprocess,
)
from day_ahead_power_forecast.ml_ops.registry import load_model
from day_ahead_power_forecast.params import BQ_DATASET, GCP_PROJECT, LOCAL_DATA_PATH

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

### app states =================================================================

# model
app.state.model = load_model()

# get preprocessed data like in main.train
query_pv = f"""
    SELECT *
    FROM {GCP_PROJECT}.{BQ_DATASET}.processed_pv
    ORDER BY local_time
"""

data_processed_pv_cache_path = Path(LOCAL_DATA_PATH).joinpath(
    "processed", "processed_pv.csv"
)
data_processed = get_data_with_cache(
    gcp_project=GCP_PROJECT,
    query=query_pv,
    cache_path=data_processed_pv_cache_path,
    data_has_header=True,
)

# the processed data from bq needs to be converted to datetime object
data_processed["local_time"] = pd.to_datetime(data_processed["local_time"], utc=True)
# rename
app.state.data_pv_clean = data_processed

### app end points =============================================================


@app.get("/predict")
def predict(
    input_date: str,
    dataset: Optional[Literal["pv", "forecast"]] = "pv",
    capacity: Optional[Literal["true", "false"]] = "false",
) -> None:
    """
    Make a prediction using the latest trained model

    Parameters
    ----------
    input_date : str
        Format: YYYY-MM-DD
        The date for which you want to make a prediction.
    dataset : str, optional
        The dataset to use for prediction. Default is 'pv'.
    capacity : str, optional
        If 'true', the prediction will be in capacity factor. Default is 'false'.
        If 'false', the prediction will be in electricity.

    Returns
    ----------
    y_pred_df : pd.DataFrame
        DataFrame containing the prediction and the corresponding UTC time.
    """

    # collect input for postprocess
    pred_df = pred(f"{input_date}")
    preprocessed_df = app.state.data_pv_clean
    if capacity == "true":
        preprocessed_df["cap_fac"] = (
            preprocessed_df.electricity / 0.9 * 100
        )  # 0.9 is max value for pv
        stats_df = get_stats_table(preprocessed_df, capacity=True)
        pred_df.pred = pred_df.pred / 0.9 * 100
    else:
        stats_df = get_stats_table(preprocessed_df, capacity=False)

    # get plot_df
    plot_df = postprocess(input_date, preprocessed_df, stats_df, pred_df)

    # Send as dict from backend to frontend; NaNs have to be replaced
    plot_df = plot_df.fillna(0.0)
    plot_dict = plot_df.to_dict()

    return plot_dict


@app.get("/")
def root():
    return {"greeting": "Hello"}
