from pathlib import Path

import pandas as pd
from colorama import Fore, Style
from google.cloud import bigquery

from day_ahead_power_forecast.params import BQ_DATASET, GCP_PROJECT
from day_ahead_power_forecast.utils import compress


def query_bq_data(
    gcp_project: str,
    query: str,
) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery
    """

    print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
    client = bigquery.Client(project=gcp_project)
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()

    print(f"✅ Data loaded, with shape {df.shape}")

    return df


def get_data_with_cache(
    gcp_project: str, query: str, cache_path: Path, data_has_header=True
) -> pd.DataFrame:
    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header="infer" if data_has_header else None)
        print(f"✅ Data loaded, with shape {df.shape}")
    else:
        df = query_bq_data(gcp_project, query)

        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)
        print("✅ Data cached locally")

    return df


def load_data_to_bq(
    data: pd.DataFrame, gcp_project: str, bq_dataset: str, table: str, truncate: bool
) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(
        Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL
    )

    # Load data onto full_table_name
    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(
        f"\n{'Write' if truncate else 'Append'} {full_table_name} \
            ({data.shape[0]} rows)"
    )

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")


def clean_forecast_data(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Initial has 3.3 M entries (everyday: 4 forecasts of 16 days ahead)
    Cleaning it to: - 1 forecast perday (at 12:00)
                    - 48 hours a day
                    - right now hardcoded to match last forecast day with
                     last day of PV data
    """
    df = compress(forecast_df)
    df = df.drop(columns=["lat", "lon", "forecast_dt_iso", "slice_dt_iso"])

    df.rename(
        columns={
            "forecast_dt_unixtime": "utc_time",
            "slice_dt_unixtime": "prediction_utc_time",
        },
        inplace=True,
    )

    # df['utc_time'] = df['utc_time'].str.replace('+0000 UTC', '')
    # df['prediction_utc_time'] = df['prediction_utc_time'].str.replace('+0000 UTC', '')

    df["utc_time"] = pd.to_datetime(df["utc_time"], unit="s", utc=True)
    df["prediction_utc_time"] = pd.to_datetime(
        df["prediction_utc_time"], unit="s", utc=True
    )

    # # get only 1 forecast per day
    df = df[df["utc_time"].dt.hour == 12]

    unique_dates = df["utc_time"].unique()

    # reduce to 24h of weather forecast (from 00:00 to 23:00 each day)
    df_revised = []
    for date in unique_dates:
        data = df[(df["utc_time"] == date)].iloc[12:36]
        df_revised.append(data)

    df_revised_ordered = pd.concat(df_revised, ignore_index=True)

    return df_revised_ordered


def get_stats_table(
    years_df: pd.DataFrame,
    capacity=False,
    min_date="2020-01-01 00:00:00",
    max_date="2022-12-29 23:00:00",
) -> pd.DataFrame:
    """
    Creates a table with statistics for electricity and optional capacity factor
    for every hour of the year (8784).
    Input:
      - Cleaned df that contains at least electricity
      as column. The df should span several years, because every
      year is one sample for the statictics.
      - Optional flag for capacity factor
    Output:
      - df with 8784 hours of the years (including leap years) as rows. The df has
      multilevel index because statistics are returned for electricity and
      capacity factor.
    """
    years_df = years_df[years_df["local_time"] < min_date]
    years_df["hour_of_year"] = years_df["local_time"].apply(
        lambda x: x.strftime("%m%d%H")
    )
    if capacity:
        stats_df = (
            years_df[["hour_of_year", "cap_fac"]]
            .groupby(["hour_of_year"])
            .agg(["mean", "median", "std", "skew", "min", "max", "count"])
        )
    else:
        stats_df = (
            years_df[["hour_of_year", "electricity"]]
            .groupby(["hour_of_year"])
            .agg(["mean", "median", "std", "skew", "min", "max", "count"])
        )
    return stats_df


def postprocess(
    today: str,
    preprocessed_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    pred_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Create a df that contains all information necessary for the plot in streamlit.
    Accumulate all data for a specific time window defined by today.
    Input:
      - today: User input from streamlit; e.g. '2000-05-15'
      - preprocessed_df: df with all years that can be selected in streamlit
        (2000-2022)
      - stats_df: provided by get_stats_table()
      - pred_df: df from pred(), should contain two columns 'utc_time' and 'pred'
    Output:
      - plot_df with columns: utc_time, local_time, electricity, hour_of_year,
        mean, median, std, skew, min, max, count, pred
      - plot_df contains NaN-values! You have to replace them for api
    """
    # define time period (3 days) for plotting
    today_timestamp = pd.Timestamp(today, tz="UTC")
    window_df = pd.date_range(
        start=today_timestamp - pd.Timedelta(days=1),
        end=today_timestamp + pd.Timedelta(days=2) - pd.Timedelta(hours=1),
        freq=pd.Timedelta(hours=1),
    ).to_frame(index=False, name="utc_time")

    # create df with the preprocessed data in the time window
    plot_df = pd.merge(
        window_df,
        preprocessed_df,
        left_on="utc_time",
        right_on="local_time",
        how="inner",
    )

    # add statistics in the time window
    plot_df["hour_of_year"] = plot_df.utc_time.apply(lambda x: x.strftime("%m%d%H"))
    stats_df.columns = stats_df.columns.droplevel(level=0)
    plot_df = pd.merge(plot_df, stats_df, on="hour_of_year", how="inner")

    # add prediction for day-ahead in time window
    if pred_df is not None:
        plot_df = pd.merge(plot_df, pred_df, on="utc_time", how="left")

    return plot_df


### ONLY FOR PROJECT INITIALIZATION  =======================================
# ==============================================================================
def get_pv_data() -> pd.DataFrame:
    """
    Load raw data from local directory and rename columns to prevent
    issues with BigQuery
    """

    csv_path = Path(__file__).parent.parent.parent.joinpath("data")
    df = pd.read_csv(csv_path.joinpath("1980-2022_pv.csv"), memory_map=True)

    df.rename(
        columns={
            "Unnamed: 0.1": "_0-1",
            "Unnamed: 0": "_0",
        },
        inplace=True,
    )

    print("# data loaded")
    return df


# Used in Makefile
def load_raw_pv() -> None:
    data_raw = get_pv_data()
    assert data_raw.columns[0] == "_0-1"
    load_data_to_bq(
        data_raw,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table="raw_pv",
        truncate=True,
    )


def get_forecast_data() -> pd.DataFrame:
    """
    Load raw data from local directory and rename columns to prevent
    issues with BigQuery
    """
    csv_path = Path(__file__).parent.parent.parent.joinpath("data")

    df = pd.read_csv(
        csv_path.joinpath("openweather_history_bulk_forecast_tempelhof.csv"),
        memory_map=True,
    )
    df.columns = ["_".join(col.split()) for col in df.columns]

    print("# data loaded")
    return df


# Used in Makefile
def load_raw_forecast() -> None:
    data_raw = get_forecast_data()
    assert data_raw.columns[0] == "forecast_dt_unixtime"
    load_data_to_bq(
        data_raw,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table="raw_weather_forecast",
        truncate=True,
    )
