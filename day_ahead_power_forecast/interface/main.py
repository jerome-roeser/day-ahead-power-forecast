from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from colorama import Fore, Style
from torch.utils.data import DataLoader
from tqdm import tqdm

from day_ahead_power_forecast.ml_ops.cross_val import (
    SequenceForecastDataset,
    SequenceGenerator,
)
from day_ahead_power_forecast.ml_ops.data import (
    clean_forecast_data,
    get_data_with_cache,
    load_data_to_bq,
    query_bq_data,
)
from day_ahead_power_forecast.ml_ops.model import RNNModel, compute_regression_metrics
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
    BQ_DATASET,
    DATASET,
    EPOCHS,
    GCP_PROJECT,
    INPUT_WIDTH,
    LABEL_WIDTH,
    LOCAL_DATA_PATH,
    MODEL_TARGET,
    SHIFT,
)


def preprocess() -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back
        from BQ during training)
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

    # stick electricity data to the forecast data
    merged_forecast_pv = pd.merge(
        data_pv_query[["local_time", "electricity"]],
        data_forecast_clean,
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


def train(
    train_start_pv: str = "1980-01-01 00:00:00",
    train_stop_pv: str = "2014-05-26 18:00:00",
    train_start_forecast: str = "2017-10-07 00:00:00",
    train_stop_forecast: str = "2021-12-13 18:00:00",
    sequences: int = 10_000,
    learning_rate: float = 0.02,
    batch_size: int = 32,
    patience: int = 5,
) -> float:
    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

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

    # Split the data into training, validating and testing sets

    n_pv = len(data_processed_pv)
    train_df_pv = data_processed_pv[0 : int(n_pv * 0.7)]
    val_df_pv = data_processed_pv[int(n_pv * 0.7) : int(n_pv * 0.9)]
    test_df_pv = data_processed_pv[int(n_pv * 0.9) :]

    sequences_pv = SequenceGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=SHIFT,
        number_sequences=10_000,
        train_df=train_df_pv,
        val_df=val_df_pv,
        test_df=test_df_pv,
        label_columns=["electricity"],
    )

    n_forecast = len(data_processed_forecast)
    train_df_forecast = data_processed_forecast[0 : int(n_forecast * 0.7 / 24) * 24]
    val_df_forecast = data_processed_forecast[
        int(n_forecast * 0.7 / 24) * 24 : int(n_forecast * 0.9 / 24) * 24
    ]
    # test_df_forecast = data_processed_forecast[int(n_forecast * 0.9 / 24) * 24 :]

    train_dataset_forecast = SequenceForecastDataset(
        df=train_df_forecast, label_columns=["electricity"]
    )
    val_dataset_forecast = SequenceForecastDataset(
        df=val_df_forecast, label_columns=["electricity"]
    )
    # test_dataset_forecast = SequenceForecastDataset(
    #     df=test_df_forecast, label_columns=["electricity"]
    # )

    if DATASET == "pv":
        train_dataset = sequences_pv.train
        val_dataset = sequences_pv.val
        # test_dataset = sequences_pv.test
        n_features = train_dataset.tensors[0].shape[-1]
    else:
        train_dataset = train_dataset_forecast
        val_dataset = val_dataset_forecast
        # test_dataset = test_dataset_forecast
        n_features = train_dataset.example[0].shape[-1]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = RNNModel(p=n_features)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    def training_one_epoch(model, train_dataloader, val_dataloader):
        size = len(train_dataloader.dataset)
        running_loss = 0
        # earlystopping = 0
        for batch, data in enumerate(train_dataloader):
            X, y = data
            output = model(X)
            loss = loss_fn(output, y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
            mae = torch.mean(abs(output - y))

            if batch % 10 == 9:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"\tloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        with torch.no_grad():
            outputs = []
            labels = []
            vsize = len(val_dataloader.dataset)
            running_vloss = 0.0

            # In evaluation mode some model specific operations can be omitted
            #  -> eg. dropout layer
            # Switching to evaluation mode, eg. turning off regularisation
            model.train(False)
            for j, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                outputs.append(voutputs)
                labels.append(vlabels)
                vloss = loss_fn(voutputs, vlabels)
                vmae = torch.mean(abs(voutputs - vlabels))
                running_vloss += vloss.item()

                if j % 10 == 9:
                    vloss, vcurrent = vloss.item(), j * batch_size + len(vinputs)
                    print(f"\tval loss: {vloss:>7f}  [{vcurrent:>5d}/{vsize:>5d}]")

            model.train(True)
        return loss, vloss, mae, vmae

    history = defaultdict(list)
    for epoch in tqdm(range(EPOCHS)):
        print(f"Epoch {epoch + 1}:")
        model.train()
        loss, vloss, mae, vmae = training_one_epoch(model, train_loader, val_loader)
        history["loss"].append(loss)
        history["val_loss"].append(vloss)
        history["mae"].append(mae)
        history["val_mae"].append(vmae)

    val_mae = np.min(history["val_mae"])

    params = dict(
        context="train",
        training_set_size=f"Training data from {train_start_forecast} to \
            {train_stop_forecast}",
        row_count=len(train_dataset),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae), history=history)

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)
    if MODEL_TARGET == "mlflow":
        mlflow_transition_model(current_stage="dev", new_stage="staging")

    print("✅ train() done \n")

    return val_mae


def evaluate(
    test_start_pv: str = "2014-05-26 19:00:00",
    test_stop_pv: str = "2022-12-30 23:00:00",
    test_start_forecast: str = "2021-12-13 18:00:00",
    test_stop_forecast: str = "2022-12-30 23:00:00",
    sequences: int = 1_000,
    batch_size: int = 32,
    stage: str = "production",
) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None
    # Query your BigQuery processed table and get data_processed using
    # `get_data_with_cache`
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

    # Split the data into training, validating and testing sets

    n_pv = len(data_processed_pv)
    train_df_pv = data_processed_pv[0 : int(n_pv * 0.7)]
    val_df_pv = data_processed_pv[int(n_pv * 0.7) : int(n_pv * 0.9)]
    test_df_pv = data_processed_pv[int(n_pv * 0.9) :]

    sequences_pv = SequenceGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=SHIFT,
        number_sequences=10_000,
        train_df=train_df_pv,
        val_df=val_df_pv,
        test_df=test_df_pv,
        label_columns=["electricity"],
    )

    n_forecast = len(data_processed_forecast)
    # train_df_forecast = data_processed_forecast[0 : int(n_forecast * 0.7 / 24) * 24]
    # val_df_forecast = data_processed_forecast[
    #     int(n_forecast * 0.7 / 24) * 24 : int(n_forecast * 0.9 / 24) * 24
    # ]
    test_df_forecast = data_processed_forecast[int(n_forecast * 0.9 / 24) * 24 :]

    # train_dataset_forecast = SequenceForecastDataset(
    #     df=train_df_forecast, label_columns=["electricity"]
    # )
    # val_dataset_forecast = SequenceForecastDataset(
    #     df=val_df_forecast, label_columns=["electricity"]
    # )
    test_dataset_forecast = SequenceForecastDataset(
        df=test_df_forecast, label_columns=["electricity"]
    )

    if DATASET == "pv":
        # train_dataset = sequences_pv.train
        # val_dataset = sequences_pv.val
        test_dataset = sequences_pv.test
        # n_features = train_dataset.tensors[0].shape[-1]
    else:
        # train_dataset = train_dataset_forecast
        # val_dataset = val_dataset_forecast
        test_dataset = test_dataset_forecast
        # n_features = train_dataset.example[0].shape[-1]

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    metrics_dict = compute_regression_metrics(model, test_loader)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate",  # Package behavior
        evaluate_set_size="10 %",
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae


def pred(input_pred: str = "2022-07-06") -> pd.DataFrame:
    """
    Make a prediction using the latest trained model
    """

    model = load_model()
    assert model is not None

    # Compute the necessary datetime objects for the BQ querying
    dt_day_ahead_begin = pd.to_datetime(input_pred, utc=True)
    dt_day_ahead_end = dt_day_ahead_begin + np.timedelta64(23, "h")

    dt_pv_data_begin = dt_day_ahead_begin - np.timedelta64(2, "D")
    dt_pv_data_end = dt_day_ahead_end - np.timedelta64(2, "D")

    dt_noon_weather_forecast = dt_day_ahead_begin - np.timedelta64(12, "h")

    print(Fore.MAGENTA + "\n⭐️ Use case: predict" + Style.RESET_ALL)

    # Query raw PV data from BUCKET BigQuery using `get_data_with_cache`
    query_pv = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.raw_pv
        WHERE _0 BETWEEN {int(dt_pv_data_begin.timestamp()) * 1000}
                    AND {int(dt_noon_weather_forecast.timestamp()) * 1000}
        ORDER BY _0
    """

    # Retrieve data from BigQuery
    data_pv_query = query_bq_data(query=query_pv, gcp_project=GCP_PROJECT)
    data_pv_query["local_time"] = pd.to_datetime(data_pv_query["local_time"], utc=True)

    # Clean data for the first 24h only
    timestamp_pv_data_end = int(dt_pv_data_end.timestamp()) * 1000
    data_pv_processed = preprocess_PV_features(
        data_pv_query.query("_0 <= @timestamp_pv_data_end")
    )

    # Query raw historical weather forecast data from BigQuery
    query_forecast = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.raw_weather_forecast
        WHERE forecast_dt_unixtime = {int(dt_noon_weather_forecast.timestamp())}
        ORDER BY forecast_dt_unixtime, slice_dt_unixtime
    """
    # AND slice_dt_unixtime BETWEEN {int(dt_day_ahead_begin.timestamp())}
    #             AND {int(dt_day_ahead_end.timestamp())}

    data_forecast_query = query_bq_data(query=query_forecast, gcp_project=GCP_PROJECT)

    # Clean data
    data_forecast_clean = clean_forecast_data(data_forecast_query)
    data_forecast_processed = preprocess_forecast_features(data_forecast_clean)

    if DATASET == "pv":
        X = data_pv_processed.select_dtypes(include=np.number)
    else:
        X = data_forecast_processed.select_dtypes(include=np.number)

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
