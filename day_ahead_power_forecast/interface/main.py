import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse
from typing import Dict, List, Tuple, Sequence
from datetime import datetime

from day_ahead_power_forecast.params import *
from day_ahead_power_forecast.ml_ops.preprocessor import preprocess_PV_features, preprocess_forecast_features
from day_ahead_power_forecast.ml_ops.data import get_data_with_cache, load_data_to_bq, clean_forecast_data
# from day_ahead_power_forecast.ml_ops.model import initialize_model, compile_model, train_model, evaluate_model
# from day_ahead_power_forecast.ml_ops.registry import load_model, save_model, save_results
# from day_ahead_power_forecast.ml_ops.cross_val import get_X_y_seq, get_X_y_seq_pv

def preprocess() -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Query raw PV data from BUCKET BigQuery using `get_data_with_cache`
    query_pv = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.raw_pv
        ORDER BY _0
    """

    # Retrieve data using `get_data_with_cache`
    data_pv_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"raw_pv.csv")
    data_pv_query = get_data_with_cache(
        query=query_pv,
        gcp_project=GCP_PROJECT,
        cache_path=data_pv_query_cache_path,
        data_has_header=True
    )
    data_pv_query['local_time'] = pd.to_datetime(data_pv_query['local_time'], utc= True)

    # Clean data
    data_pv_processed = preprocess_PV_features(data_pv_query)

    load_data_to_bq(
        data_pv_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_pv',
        truncate=True
    )

     # Query raw historical weather forecast data from BUCKET BigQuery
     # using `get_data_with_cache`
    query_forecast = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.raw_weather_forecast
        ORDER BY forecast_dt_unixtime, slice_dt_unixtime
    """

    # Retrieve data using `get_data_with_cache`
    data_forecast_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"raw_weather_forecast.csv")
    data_forecast_query = get_data_with_cache(
        query=query_forecast,
        gcp_project=GCP_PROJECT,
        cache_path=data_forecast_query_cache_path,
        data_has_header=True
    )

    # Clean data
    data_forecast_clean = clean_forecast_data(data_forecast_query)

    # stick electricity data to the forecast data
    merged_forecast_pv = pd.merge(\
        data_pv_query[['local_time', 'electricity']], \
            data_forecast_clean, \
                left_on='local_time', \
                    right_on= 'prediction_utc_time', \
                        how='inner')

    data_forecast_processed = preprocess_forecast_features(merged_forecast_pv)

    load_data_to_bq(
        data_forecast_processed,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_weather_forecast',
        truncate=True
    )

    print("✅ preprocess() done \n")


def train(
        train_start_pv: str = '1980-01-01 00:00:00',
        train_stop_pv: str = '2014-05-26 18:00:00',
        train_start_forecast: str = '2017-10-07 00:00:00',
        train_stop_forecast: str = '2021-12-13 18:00:00',
        sequences: int = 10_000,
        learning_rate: float =0.02,
        batch_size: int = 32,
        patience: int = 5,
        forecast_features: bool = False
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed training data..." + Style.RESET_ALL)


    # --First-- Load processed PV data using `get_data_with_cache` in chronological order
    query_pv = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_pv
        ORDER BY local_time
    """

    data_processed_pv_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_pv.csv")
    data_processed_pv = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query_pv,
        cache_path=data_processed_pv_cache_path,
        data_has_header=True
    ).select_dtypes(include=np.number)


    if data_processed_pv.shape[0] < 240:
        print("❌ Not enough processed data retrieved to train on")
        return None


    query_forecast = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_weather_forecast
        ORDER BY utc_time, prediction_utc_time
    """

    data_processed_forecast_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_weather_forecast.csv")
    data_processed_forecast = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query_forecast,
        cache_path=data_processed_forecast_cache_path,
        data_has_header=True
    ).select_dtypes(include=np.number)

    if data_processed_forecast.shape[0] < 240:
        print("❌ Not enough processed data retrieved to train on")
        return None

    # Split the data into training and testing sets

    n_pv = len(data_processed_pv)
    train_df_pv = data_processed_pv[0:int(n_pv*0.7)]
    val_df_pv = data_processed_pv[int(n_pv*0.7):int(n*0.9)]
    test_df_pv = data_processed_pv[int(n_pv*0.9):]

    sequences_pv = SequenceGenerator(input_width= INPUT_WIDTH,
                              label_width= LABEL_WIDTH,
                              shift= SHIFT,
                              number_sequences= 10_000,
                              train_df= train_df_pv,
                              val_df= val_df_pv,
                              test_df= test_df_pv,
                              label_columns= ['electricity'])

    train_dataset_pv = sequences_pv.train
    val_dataset_pv = sequences_pv.val
    test_dataset_pv = sequences_pv.test


    n_forecast = len(data_processed_forecast)
    train_df_forecast = data_processed_forecast[0:int(n_forecast*0.7 / 24) * 24]
    val_df_forecast = data_processed_forecast[int(n_forecast*0.7 / 24) * 24:int(n*0.9 / 24) * 24]
    test_df_forecast = data_processed_forecast[int(n_forecast*0.9 / 24) * 24:]

    train_dataset_forecast = SequenceForecastDataset(df= train_df_forecast, label_columns=['electricity'])
    val_dataset_forecast = SequenceForecastDataset(df= val_df_forecast, label_columns=['electricity'])
    test_dataset_forecast = SequenceForecastDataset(df= test_df_forecast, label_columns=['electricity'])

    if dataset == 'pv':
        train_dataset = train_dataset_pv
        val_dataset = val_dataset_pv
        test_dataset = test_dataset_pv
        n_features = train_dataset_pv.tensors[0].shape[-1]
    else:
        train_dataset = train_dataset_forecast
        val_dataset = val_dataset_forecast
        test_dataset = test_dataset_forecast
        n_features = train_dataset.example[0].shape[-1]

    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle= True)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle= True)

    model = RNNModel(p = n_features)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(params= model.parameters(), lr= learning_rate)

    def training_one_epoch(model, train_dataloader, val_dataloader):
        size = len(train_dataloader.dataset)
        running_loss = 0
        earlystopping = 0
        for batch, data in enumerate(train_dataloader):
            X, y = data
            output = model(X)
            loss = loss_fn(output, y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()

            if batch % 10 == 9:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"\tloss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        with torch.no_grad():
            outputs = []
            labels = []
            vsize = len(val_dataloader.dataset)
            running_vloss = 0.0

            # In evaluation mode some model specific operations can be omitted eg. dropout layer
            model.train(False) # Switching to evaluation mode, eg. turning off regularisation
            for j, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()

                if j % 10 == 9:
                    vloss, vcurrent = vloss.item(), j * batch_size + len(vinputs)
                    print(f"\tval loss: {vloss:>7f}  [{vcurrent:>5d}/{vsize:>5d}]")

            model.train(True)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}:')
        model.train()
        training_one_epoch(model, train_loader, val_loader)

    metrics = compute_regression_metrics(model, test_loader)

    # Train model using `model.py`
    model = load_model(forecast_features=True)

    if model is None:
        model = initialize_model(X_train, y_train, n_unit=24)

    model = initialize_model(X_train, y_train, n_unit=24)

    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(model,
                                X_train,
                                y_train,
                                validation_split = 0.3,
                                batch_size = 32,
                                epochs = 50
                                )
    val_mae = np.min(history.history['val_mae'])

    params = dict(
        context="train",
        training_set_size=f'Training data from {train_start_forecast} to {train_stop_forecast}',
        row_count=len(X_train),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae), history=history)

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model, forecast_features= True)

    print("✅ train() done \n")

    return val_mae


def evaluate(
        test_start_pv: str = '2014-05-26 19:00:00',
        test_stop_pv: str = '2022-12-30 23:00:00',
        test_start_forecast: str = '2021-12-13 18:00:00',
        test_stop_forecast: str = '2022-12-30 23:00:00',
        sequences: int = 1_000,
        forecast_features: bool = True,
        stage: str = "Production"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)



    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_pv
        ORDER BY utc_time
    """

    data_processed_pv_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_pv.csv")
    data_processed_pv = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_pv_cache_path,
        data_has_header=True
    )

    # the processed PV data from bq needs to be converted to datetime object
    data_processed_pv.local_time = pd.to_datetime(data_processed_pv.local_time,utc=True)

    if data_processed_pv.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    if forecast_features:
    # --Second-- Load processed Weather Forecast data in chronological order
        query_forecast = f"""
            SELECT *
            FROM {GCP_PROJECT}.{BQ_DATASET}.processed_weather_forecast
            ORDER BY utc_time, predicition_utc_time
        """

        data_processed_forecast_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_weather_forecast.csv")
        data_processed_forecast = get_data_with_cache(
            gcp_project=GCP_PROJECT,
            query=query_forecast,
            cache_path=data_processed_forecast_cache_path,
            data_has_header=True
        )

        if data_processed_forecast.shape[0] < 240:
            print("❌ Not enough processed data retrieved to train on")
            return None

        # Split the data into training and testing sets
        test_pv = data_processed_pv[(data_processed_pv['utc_time'] > test_start_forecast)\
                    & (data_processed_pv['utc_time'] < test_stop_forecast)]
        test_forecast = data_processed_forecast[data_processed_forecast['utc_time'] > test_start_forecast]

        X_test, y_test = get_X_y_seq(test_pv,
                                    test_forecast,
                                    number_of_sequences= sequences,
                                    input_length=48,
                                    output_length=24,
                                    gap_hours=12)
        model = load_model(forecast_features= True, stage=stage)
        assert model is not None

        metrics_dict = evaluate_model(model=model, X=X_test, y=y_test)
        mae = metrics_dict["mae"]

    else:
        # Split the data into training and testing sets
        test_pv = data_processed_pv[data_processed_pv['utc_time'] > test_start_pv]

        X_test, y_test = get_X_y_seq_pv(test_pv,
                                    number_of_sequences= sequences,
                                    input_length=48,
                                    output_length=24,
                                    gap_hours=12)

        model = load_model(stage=stage)
        assert model is not None
        metrics_dict = evaluate_model(model=model, X=X_test, y=y_test)
        mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
        evaluate_set_size="3 years",
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae


def pred(input_pred:str = '2022-07-06 12:00:00',
         forecast_features: bool = False) -> pd.DataFrame:
    """
    Make a prediction using the latest trained model
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: predict" + Style.RESET_ALL)

    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_pv
        ORDER BY utc_time
    """

    data_processed_pv_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_pv.csv")
    data_processed_pv = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_pv_cache_path,
        data_has_header=True
    )

    # X_pred should be the 48 hours before the input date
    X_pred_pv = data_processed_pv[data_processed_pv['utc_time'] < input_pred][-48:]

    if forecast_features:
    # --Second-- Load processed Weather Forecast data in chronological order
        query_forecast = f"""
            SELECT *
            FROM {GCP_PROJECT}.{BQ_DATASET}.processed_weather_forecast
            ORDER BY utc_time, predicition_utc_time
        """

        data_processed_forecast_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_weather_forecast.csv")
        data_processed_forecast = get_data_with_cache(
            gcp_project=GCP_PROJECT,
            query=query_forecast,
            cache_path=data_processed_forecast_cache_path,
            data_has_header=True
        )

        if data_processed_forecast.shape[0] < 240:
            print("❌ Not enough processed data retrieved to train on")
            return None

        input_date = input_pred.split()[0]
        X_pred_forecast = get_weather_forecast_features(data_processed_forecast, input_date)

        X_pred_pv = X_pred_pv.reset_index()
        to_concat = [X_pred_pv.iloc[:, 2:], X_pred_forecast.iloc[:, 2:]]

        X_pred = pd.concat(to_concat, axis=1)
        feature_indices = {name:i for i, name in enumerate(X_pred)}
        X_pred = X_pred.to_numpy()
        X_pred_tf = tf.convert_to_tensor(X_pred)
        X_pred_tf = tf.expand_dims(X_pred_tf, axis=0)

        model = load_model(forecast_features= True)
        assert model is not None

    else:
        # convert X_pred to a tensorflow object
        X_pred = X_pred_pv.iloc[:, 1:]
        feature_indices = {name:i for i, name in enumerate(X_pred)}
        X_pred = X_pred.to_numpy()
        X_pred_tf = tf.convert_to_tensor(X_pred)
        X_pred_tf = tf.expand_dims(X_pred_tf, axis=0)

        model = load_model()
        assert model is not None



    print(Fore.BLUE + f"\nPredict with {feature_indices} X_pred tensors \
        \n -> forecast features: {forecast_features}" + Style.RESET_ALL)
    y_pred = model.predict(X_pred_tf)

    # y_pred dates shoud be the 24hours after a 12 hour gap
    y_pred_df = data_processed_pv[data_processed_pv['utc_time'] > input_pred][12:36]

    y_pred_df['pred'] = y_pred[0]

    # y_pred_df should have only two columns: 'utc_time', 'pred'; utc_time
    # should be datetime object

    y_pred_df = y_pred_df[['utc_time', 'pred']]
    y_pred_df.reset_index(drop=True, inplace=True)
    y_pred_df.local_time = pd.to_datetime(y_pred_df.local_time,utc=True)

    # Cut-off predictions that are negative or bigger than max capacity
    def cutoff_func(x):
      if x < 0.0:
        return 0
      if x > 0.9:
        return 0.9
      return x

    y_pred_df.pred = y_pred_df.pred.apply(cutoff_func)


    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")

    return y_pred_df


if __name__ == '__main__':
    preprocess()
    train()
    evaluate()
    pred()
