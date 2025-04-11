from collections import defaultdict
from pathlib import Path
from typing import Literal, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from colorama import Fore, Style
from mlflow.models import infer_signature
from torch.utils.data import DataLoader
from tqdm import tqdm

from day_ahead_power_forecast.ml_ops.cross_val import (
    PhotovoltaicDataWindowGenerator,
    WeatherForecastDataset,
)
from day_ahead_power_forecast.ml_ops.data import (
    clean_forecast_data,
    get_data_with_cache,
    load_data_to_bq,
    query_bq_data,
)
from day_ahead_power_forecast.ml_ops.model import (
    EarlyStopper,
    LSTMModel_2,
    compute_regression_metrics,
    training_one_epoch,
)
from day_ahead_power_forecast.ml_ops.preprocessor import (
    preprocess_forecast_features,
    preprocess_PV_features,
)
from day_ahead_power_forecast.ml_ops.registry import (
    load_model,
    mlflow_transition_model,
    save_model,
    save_results,
)
from day_ahead_power_forecast.params import (
    BATCH_SIZE,
    BQ_DATASET,
    DATASET,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    GCP_PROJECT,
    INPUT_WIDTH,
    LABEL_WIDTH,
    LEARNING_RATE,
    LOCAL_DATA_PATH,
    MLFLOW_EXPERIMENT,
    MODEL_TARGET,
    SHIFT,
)


def preprocess() -> None:
    """
    Prepare the data for the training and evaluation of the model.
        - Query the raw dataset from BigQuery
        - Cache query result as a local CSV if it doesn't exist locally
        - Clean the the historical PV production data and encode the time features
        - Adapt the wheather forecast data to 24h of prediction of the day ahead
        - add the PV production label to the data as well as a PV production feature
        - Store processed data on BigQuery
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Query raw PV data from BUCKET BigQuery using `get_data_with_cache`
    query_pv = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.raw_pv
        ORDER BY _0
    """

    # Retrieve data using `get_data_with_cache`
    data_pv_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", "raw_pv.csv")
    data_pv_query = get_data_with_cache(
        query=query_pv,
        gcp_project=GCP_PROJECT,
        cache_path=data_pv_query_cache_path,
        data_has_header=True,
    )
    data_pv_query["local_time"] = pd.to_datetime(data_pv_query["local_time"], utc=True)

    # Clean data
    data_pv_processed = preprocess_PV_features(data_pv_query)

    load_data_to_bq(
        data_pv_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table="processed_pv",
        truncate=True,
    )

    # Query raw historical weather forecast data from BUCKET BigQuery
    # using `get_data_with_cache`
    query_forecast = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.raw_weather_forecast
        ORDER BY forecast_dt_unixtime, slice_dt_unixtime
    """

    # Retrieve data using `get_data_with_cache`
    data_forecast_query_cache_path = Path(LOCAL_DATA_PATH).joinpath(
        "raw", "raw_weather_forecast.csv"
    )
    data_forecast_query = get_data_with_cache(
        query=query_forecast,
        gcp_project=GCP_PROJECT,
        cache_path=data_forecast_query_cache_path,
        data_has_header=True,
    )

    # Clean data
    data_forecast_clean = clean_forecast_data(data_forecast_query)

    # stick day_ahead (label) and day_before (feature) electricity data to the
    # weather forecast data

    # the electricity feature is the electricity produced 2 days before
    # the production amount we want to predict
    data_forecast_clean["electricity_feature_utc_time"] = data_forecast_clean[
        "prediction_utc_time"
    ] - np.timedelta64(2, "D")

    merged_forecast_pv = pd.merge(
        data_pv_query[["local_time", "electricity"]],
        data_forecast_clean,
        left_on="local_time",
        right_on="electricity_feature_utc_time",
        how="inner",
    )
    merged_forecast_pv = merged_forecast_pv.rename(
        columns={"electricity": "electricity_feature"}
    )

    merged_forecast_pv = pd.merge(
        data_pv_query[["local_time", "electricity"]],
        merged_forecast_pv,
        left_on="local_time",
        right_on="prediction_utc_time",
        how="inner",
    )

    data_forecast_processed = preprocess_forecast_features(merged_forecast_pv)

    load_data_to_bq(
        data_forecast_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table="processed_weather_forecast",
        truncate=True,
    )

    print("✅ preprocess() done \n")


def train(train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1)) -> float:
    """
    Train the model on the processed data
        - Download processed data from your BQ table (or from cache if it exists)
        - Train on the preprocessed dataset
        - Store training results and model weights

    Parameters
    ----------
    train_val_test_split : tuple(float, float, float)
        The split of the dataset into train, validation and test set
        The default is (0.7, 0.2, 0.1).

    Returns
    -------
    val_mae: float
        The mean absolute error of the model on the validation set
    """
    ############# Query Data ###############################################
    # ==========================================================================
    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed training data..." + Style.RESET_ALL)

    query_pv = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_pv
        ORDER BY local_time
    """

    data_processed_pv_cache_path = Path(LOCAL_DATA_PATH).joinpath(
        "processed", "processed_pv.csv"
    )
    data_processed_pv = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query_pv,
        cache_path=data_processed_pv_cache_path,
        data_has_header=True,
    ).select_dtypes(include=np.number)

    if data_processed_pv.shape[0] < 240:
        print("❌ Not enough processed data retrieved to train on")
        return None

    query_forecast = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_weather_forecast
        ORDER BY utc_time, prediction_utc_time
    """

    data_processed_forecast_cache_path = Path(LOCAL_DATA_PATH).joinpath(
        "processed", "processed_weather_forecast.csv"
    )
    data_processed_forecast = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query_forecast,
        cache_path=data_processed_forecast_cache_path,
        data_has_header=True,
    ).select_dtypes(include=np.number)

    if data_processed_forecast.shape[0] < 240:
        print("❌ Not enough processed data retrieved to train on")
        return None

    ############# Split the data into training & validating sets ###############
    # =========================================================================
    train_ratio, val_ratio, test_ratio = train_val_test_split

    n_pv = len(data_processed_pv)
    train_df_pv = data_processed_pv[0 : int(n_pv * train_ratio)]
    val_df_pv = data_processed_pv[
        int(n_pv * train_ratio) : int(n_pv * (train_ratio + val_ratio))
    ]

    sequences_pv = PhotovoltaicDataWindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=SHIFT,
        number_sequences=10_000,
        train_df=train_df_pv,
        val_df=val_df_pv,
        label_columns=["electricity"],
    )

    try:
        assert train_ratio + val_ratio + test_ratio == 1
    except AssertionError:
        print("❌ The sum of the train, validation and test ratios must be equal to 1.")
        return None

    n_forecast = len(data_processed_forecast)
    train_df_forecast = data_processed_forecast[
        0 : int(n_forecast * train_ratio / 24) * 24
    ]
    val_df_forecast = data_processed_forecast[
        int(n_forecast * 0.7 / 24) * 24 : int(
            n_forecast * (train_ratio + val_ratio) / 24
        )
        * 24
    ]

    train_dataset_forecast = WeatherForecastDataset(
        df=train_df_forecast, label_columns=["electricity"]
    )
    val_dataset_forecast = WeatherForecastDataset(
        df=val_df_forecast, label_columns=["electricity"]
    )

    if DATASET == "pv":
        train_dataset = sequences_pv.train
        val_dataset = sequences_pv.val
        n_features = train_dataset.tensors[0].shape[-1]
    else:
        train_dataset = train_dataset_forecast
        val_dataset = val_dataset_forecast
        n_features = train_dataset.example[0].shape[-1]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ########### Train the model ###############################################
    # ==========================================================================
    model = LSTMModel_2(p=n_features)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    history = defaultdict(list)
    early_stopping = EarlyStopper(patience=EARLY_STOPPING_PATIENCE)
    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

    for epoch in tqdm(range(EPOCHS)):
        print(f"Epoch {epoch + 1}:")
        model.train()
        loss, vloss, mae, vmae = training_one_epoch(
            model,
            train_loader,
            val_loader,
            loss_fn=loss_fn,
            optimizer=optim,
            batch_size=BATCH_SIZE,
        )

        history["loss"].append(loss)
        history["val_loss"].append(vloss)
        history["mae"].append(mae)
        history["val_mae"].append(vmae)

        if early_stopping.early_stop(vloss):
            print(f"Early stopping at epoch {epoch}")
            break

    index_min_mae = np.argmin(history["val_mae"])
    index_min_vloss = -EARLY_STOPPING_PATIENCE
    train_metrics = {
        "loss": history["loss"][index_min_vloss].detach(),
        "val_loss": history["val_loss"][index_min_vloss],
        "mae": history["mae"][index_min_vloss].detach(),
        "val_mae": history["val_mae"][index_min_vloss],
    }

    params = {
        "context": "train",
        "dataset": DATASET,
        "training_set_size": f"{(1 - test_ratio) * 100} %",
        "epochs": EPOCHS,
        "patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": "MeanAbsoluteError",
        "optimizer": optim.__class__.__name__,
    }

    # Save training results and model summary
    save_results(params=params, metrics=train_metrics, history=history)

    # Save model weights & summary
    if DATASET == "pv":
        signature = infer_signature(
            model_input=train_dataset[0][0].numpy(),
            model_output=train_dataset[0][1].numpy(),
        )
    else:
        signature = infer_signature(
            model_input=train_dataset[0][0],
            model_output=train_dataset[0][1],
        )

    save_model(model=model, signature=signature)
    if MODEL_TARGET == "mlflow":
        mlflow_transition_model(current_stage="dev", new_stage="staging")

    print("✅ train() done \n")

    return train_metrics["val_mae"]


def evaluate(
    stage: Literal["dev", "staging", "production", "archived"] = "production",
    split_ratio: float = 0.1,
) -> float:
    """
    Evaluate the performance of the latest production model on processed data

    Parameters
    ----------
    stage : str
        Stage of the model to load (e.g., "production", "staging")
    split_ratio : float
        The ratio of the dataset to use for evaluation (default is 0.1)

    Returns
    -------
    mae : float
        Mean Absolute Error of the model on the test set
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None

    ############# Query Data ###############################################
    # ==========================================================================
    query_pv = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_pv
        ORDER BY local_time
    """

    data_processed_pv_cache_path = Path(LOCAL_DATA_PATH).joinpath(
        "processed", "processed_pv.csv"
    )
    data_processed_pv = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query_pv,
        cache_path=data_processed_pv_cache_path,
        data_has_header=True,
    ).select_dtypes(include=np.number)

    if data_processed_pv.shape[0] < 240:
        print("❌ Not enough processed data retrieved to train on")
        return None

    query_forecast = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_weather_forecast
        ORDER BY utc_time, prediction_utc_time
    """

    data_processed_forecast_cache_path = Path(LOCAL_DATA_PATH).joinpath(
        "processed", "processed_weather_forecast.csv"
    )
    data_processed_forecast = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query_forecast,
        cache_path=data_processed_forecast_cache_path,
        data_has_header=True,
    ).select_dtypes(include=np.number)

    if data_processed_forecast.shape[0] < 240:
        print("❌ Not enough processed data retrieved to train on")
        return None

    ############# Prepare the test set #######################################
    # =========================================================================

    n_pv = len(data_processed_pv)
    train_df_pv = data_processed_pv[0 : int(n_pv * (1 - split_ratio))]
    test_df_pv = data_processed_pv[int(n_pv * (1 - split_ratio)) :]

    sequences_pv = PhotovoltaicDataWindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=SHIFT,
        number_sequences=10_000,
        train_df=train_df_pv,
        test_df=test_df_pv,
        label_columns=["electricity"],
    )

    n_forecast = len(data_processed_forecast)
    test_df_forecast = data_processed_forecast[
        int(n_forecast * (1 - split_ratio) / 24) * 24 :
    ]

    test_dataset_forecast = WeatherForecastDataset(
        df=test_df_forecast, label_columns=["electricity"]
    )

    if DATASET == "pv":
        test_dataset = sequences_pv.test
    else:
        test_dataset = test_dataset_forecast

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ############# Evaluate the model ##########################################
    # ==========================================================================

    metrics_dict = compute_regression_metrics(model, test_loader)
    mae = metrics_dict["mae"]

    params = {
        "context": "evaluate",
        "evaluate_set_size": f"{split_ratio * 100} %",
    }

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae


def pred(input_pred: str = "2022-07-06") -> pd.DataFrame:
    """
    Make a prediction using the latest trained model

    Parameters
    ----------
    input_pred : str
        Format: ***YYYY-MM-DD***
        The date for which you want to make a prediction.

    Returns
    ----------
    y_pred_df : pd.DataFrame
        DataFrame containing the prediction and the corresponding UTC time.
    """

    model = load_model()
    assert model is not None

    # Compute the necessary datetime objects for the BQ querying
    dt_day_ahead_begin = pd.to_datetime(input_pred, utc=True)
    dt_day_ahead_end = dt_day_ahead_begin + np.timedelta64(23, "h")

    dt_noon_weather_forecast = dt_day_ahead_begin - np.timedelta64(12, "h")

    dt_pv_data_begin = dt_noon_weather_forecast - np.timedelta64(47, "h")
    dt_pv_data_end = dt_noon_weather_forecast

    print(Fore.MAGENTA + "\n⭐️ Use case: predict" + Style.RESET_ALL)

    ############# Query Data ###############################################
    # ==========================================================================
    query_pv = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.raw_pv
        WHERE _0 BETWEEN {int(dt_pv_data_begin.timestamp()) * 1000}
                    AND {int(dt_pv_data_end.timestamp()) * 1000}
        ORDER BY _0
    """

    # Retrieve data from BigQuery
    data_pv_query = query_bq_data(query=query_pv, gcp_project=GCP_PROJECT)
    data_pv_query["local_time"] = pd.to_datetime(data_pv_query["local_time"], utc=True)

    # Clean data
    data_pv_processed = preprocess_PV_features(data_pv_query)

    # Query raw historical weather forecast data from BigQuery
    query_forecast = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.raw_weather_forecast
        WHERE forecast_dt_unixtime = {int(dt_noon_weather_forecast.timestamp())}
        ORDER BY forecast_dt_unixtime, slice_dt_unixtime
    """

    data_forecast_query = query_bq_data(query=query_forecast, gcp_project=GCP_PROJECT)

    # Clean data
    data_forecast_clean = clean_forecast_data(data_forecast_query)

    # stick day_ahead (label) electricity data to the weather forecast data

    # the electricity feature is the electricity produced 2 days before
    # the production amount we want to predict
    data_forecast_clean["electricity_feature_utc_time"] = data_forecast_clean[
        "prediction_utc_time"
    ] - np.timedelta64(2, "D")

    merged_forecast_pv = pd.merge(
        data_pv_query[["local_time", "electricity"]],
        data_forecast_clean,
        left_on="local_time",
        right_on="electricity_feature_utc_time",
        how="inner",
    )
    merged_forecast_pv = merged_forecast_pv.rename(
        columns={"electricity": "electricity_feature"}
    )

    data_forecast_processed = preprocess_forecast_features(merged_forecast_pv)

    if DATASET == "pv":
        X = data_pv_processed.select_dtypes(include=np.number)
    else:
        X = data_forecast_processed.select_dtypes(include=np.number)

    ############# Make prediction ##############################################
    # ==========================================================================

    X = np.expand_dims(X.values, axis=0)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model.eval()
    y_pred = model.forward(X_tensor)

    y_pred_df = pd.DataFrame(y_pred.detach().numpy()[0], columns=["pred"])
    y_pred_df["utc_time"] = data_forecast_clean["prediction_utc_time"]

    # Cut-off predictions that are negative or bigger than max capacity
    def cutoff_func(x: float, max_capacity: float = 0.9) -> float:
        """
        Cut off the prediction to be between 0 and max_capacity
        """
        return np.minimum(x, max_capacity) * (x > 0)

    y_pred_df["pred"] = y_pred_df["pred"].apply(cutoff_func)

    print("\n✅ prediction done: ", y_pred_df, y_pred_df.shape, "\n")

    return y_pred_df


if __name__ == "__main__":
    preprocess()
    train()
    evaluate()
    pred()
