import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from day_ahead_power_forecast.ml_ops.data import get_data_with_cache, get_stats_table, postprocess
from day_ahead_power_forecast.ml_ops.registry import load_model
from day_ahead_power_forecast.interface.main import pred #, postprocess

from pathlib import Path
from day_ahead_power_forecast.params import *
from day_ahead_power_forecast.utils import compress

# Request URLs:
# http://127.0.0.1:8000/extract_data?input_date=2018-07-06%2000%3A00%3A00&n_days=10&power_source=pv
# http://127.0.0.1:8000/baseline_yesterday?input_date=2019-07-06%2000%3A00%3A00&n_days=10&power_source=pv
# http://127.0.0.1:8000/predict?input_date=2020-07-06%2000%3A00%3A00&n_days=10&power_source=pv


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
data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_pv.csv")
query = f"""
    SELECT *
    FROM {GCP_PROJECT}.{BQ_DATASET}.processed_pv
    ORDER BY utc_time
"""

data_processed = get_data_with_cache(
    gcp_project=GCP_PROJECT,
    query=query,
    cache_path=data_processed_cache_path,
    data_has_header=True
)

# the model uses power as feature -> fix that in raw data
#data_processed = data_processed.rename(columns={'electricity': 'power'})
# the processed data from bq needs to be converted to datetime object
data_processed.utc_time = pd.to_datetime(data_processed.utc_time,utc=True)
# rename
app.state.data_pv_clean = data_processed

### app end points =============================================================

@app.get("/visualisation")
def visualisation(input_date: str, power_source='pv', capacity='false') -> None:
  """
  input_date corresponds to "today"
  """

  # collect input for postprocess
  pred_df = pred( f"{input_date} 12:00:00")
  preprocessed_df = app.state.data_pv_clean
  if capacity == 'true':
    print('Capacity!')
    preprocessed_df['cap_fac'] = preprocessed_df.electricity / 0.9 * 100 # 0.9 is max value for pv
    stats_df = get_stats_table(preprocessed_df, capacity=True)
    pred_df.pred = pred_df.pred / 0.9 * 100
  else:
    print('Electricity!')
    stats_df = get_stats_table(preprocessed_df, capacity=False)

  # get plot_df
  plot_df = postprocess(input_date, preprocessed_df, stats_df, pred_df)

  # Send as dict from backend to frontend; NaNs have to be replaced
  plot_df = plot_df.fillna(0.0)
  plot_dict = plot_df.to_dict()

  return plot_dict


@app.get("/extract_data")
def extract_pv_data(input_date: str, n_days=10, power_source='pv'):
    data_pv_clean = app.state.data_pv_clean

    n_rows = 24 * int(n_days)
    df_before = data_pv_clean[data_pv_clean['utc_time'] < input_date][-n_rows:]
    days_before = {
        'date':df_before.utc_time.to_list(),
        'power_source':df_before.electricity.to_list()
        }

    df_after = data_pv_clean[data_pv_clean['utc_time'] >= input_date][:24]
    day_after = {
        'date':df_after.utc_time.to_list(),
        'power_source':df_after.electricity.to_list()
        }

    return {input_date: {'days_before':days_before, 'day_after':day_after}}

@app.get("/baseline_yesterday")
def predict_baseline_yesterday(input_date: str, n_days=0, power_source='pv'):
    data_pv_clean = app.state.data_pv_clean
    data = data_pv_clean[data_pv_clean['utc_time'] < input_date][-24:]
    baseline_data = {
        'date':data.utc_time.to_list(),
        'power_source':data.electricity.to_list()
        }
    return {input_date: baseline_data}

@app.get("/predict")
def predict(input_date: str, n_days=2, power_source='pv'):
    pv_data_clean = app.state.data_pv_clean
    X_pred = pv_data_clean[pv_data_clean['utc_time'] < input_date][-48:]

    # model = app.state.model
    # assert model is not None

    # y_pred = model.predict(df)

    predicted_data = {
        'date':X_pred.utc_time.to_list(),
        'power_source':X_pred.electricity.to_list()
        }

    return {f'dataframe to predict': X_pred.electricity.to_list()}


@app.get("/")
def root():

    return {'greeting': 'Hello'}

#################### OLD ENDPOINT   ########################

# @app.get("/predict/huajie")
# def predict(
#     given_date: str,  # 2013-07-06
#     ):

#     df= pd.read_csv('raw_data/1980-2022_pv.csv')
#     given_date_converted = datetime.strptime(given_date, "%Y-%m-%d")
#     one_day_before = given_date_converted - timedelta(days=1)
#     given_date_before = one_day_before.strftime("%Y-%m-%d")
#     given_date_dt = pd.to_datetime(given_date_before)
#     df['local_time'] = pd.to_datetime(df['local_time'], utc=True)
#     selected_data = df[df['local_time'].dt.date == pd.to_datetime(given_date_dt).date()]
#     new_df = pd.DataFrame({
#         'local_time': selected_data['local_time'].dt.strftime('%Y-%m-%d %H:%M:%S'),
#         'electricity': selected_data['electricity']
#     })
#     df_dict = new_df.to_dict(orient='records')
#     converted_dict = {entry['local_time']: entry['electricity'] for entry in df_dict}

#     return converted_dict
