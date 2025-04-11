from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm


class PhotovoltaicDataWindowGenerator:
    """
    Data windowing class for PV time series data.

    This class is used to create sequences of data for training and testing
    machine learning models and convert them to a pytorch Dataset.

    This class can:
        - Handle the indexes and offsets for data windowing.
        - Split windows of features into (features, labels) pairs.
        - Efficiently generate batches of these windows from the training,
        evaluation, and test data, using pytorch's TensorDataset class.


    Parameters
    ----------
    input_width: int
        Length of the input sequence
    label_width: int
        Length of the label sequence
    shift: int
        Number of time steps to shift the window (label width + time gap)
        e.g. if label width is 24h and time gap  between input and prediction is 12h, shift = 36
    number_sequences: int
        Number of sequences to create
    train_df: pd.DataFrame
        DataFrame containing the training data
    val_df: pd.DataFrame
        DataFrame containing the validation data
    test_df: pd.DataFrame
        DataFrame containing the test data
    label_columns: List[str]
        List of column names to be used as labels

    Attributes
    ----------
    column_indices: dict
        Dictionary mapping column names to indices
    label_columns_indices: dict
        Dictionary mapping label column names to indices
    input_slice: slice
        Slice object for the input data
    labels_slice: slice
        Slice object for the labels
    input_indices: np.ndarray
        Array of input indices
    label_indices: np.ndarray
        Array of label indices
    total_window_size: int
        Total size of the window
    input_width: int
        Length of the input sequence
    label_width: int
        Length of the label sequence
    shift: int
        Number of time steps to shift the window
    number_of_sequences: int
        Number of sequences to create
    train_df: pd.DataFrame
        DataFrame containing the training data
    val_df: pd.DataFrame
        DataFrame containing the validation data
    test_df: pd.DataFrame
        DataFrame containing the test data
    """

    def __init__(
        self,
        input_width: int,
        label_width: int,
        shift: int,
        number_sequences: int,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        label_columns: Optional[List[str]] = None,
    ):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.number_of_sequences = number_sequences
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, self.total_window_size)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return "\n".join(
            [
                f"Number of sequences: {self.number_of_sequences}",
                f"Total window size: {self.total_window_size}",
                f"Input indices: {self.input_indices}",
                f"Label indices: {self.label_indices}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    def split_window(self, features):
        inputs = features[self.input_slice, :]
        labels = features[self.labels_slice, :]
        if self.label_columns is not None:
            labels = np.stack(
                [labels[:, self.column_indices[name]] for name in self.label_columns],
                axis=-1,
            )

        return inputs, labels

    def make_dataset(self, data: pd.DataFrame):
        "Create a dataset of x sequences of features and labels"
        data = np.array(data)
        last_full_sequence_start = len(data) - self.total_window_size
        inputs, labels = [], []

        for n in tqdm(range(self.number_of_sequences)):
            random_start = np.random.randint(0, last_full_sequence_start)
            input, label = self.split_window(data[random_start:])

            inputs.append(torch.tensor(input, dtype=torch.float32))
            labels.append(torch.tensor(label, dtype=torch.float32))

        inputs = torch.cat(inputs).view(self.number_of_sequences, self.input_width, -1)
        labels = torch.cat(labels).view(self.number_of_sequences, self.label_width, -1)
        return TensorDataset(inputs, labels)

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


class WeatherForecastDataset(Dataset):
    """
    Dataset class the Weather Forecast time series data.

    Parameters
    ----------
    df: pd.DataFrame
        Weather Forecast DataFrame
    number_days_forecast: int
        Number of days to forecast
    label_columns: List[str]
        List of column names to be used as labels

    Attributes
    ----------
    df: pd.DataFrame
        DataFrame containing the data
    forecast_hours: int
        Number of hours to forecast
    number_of_sequences: int
        Number of
    label_columns: List[str]
        List of column names to be used as labels
    label_columns_indices: dict
        Dictionary mapping label column names to indices
    column_indices: dict
        Dictionary mapping column names to indices
    feature_columns: List[str]
        List of column names to be used as features
    """

    def __init__(
        self,
        df: pd.DataFrame,
        number_days_forecast: int = 1,
        label_columns: List[str] = None,
    ):
        self.df = df.copy().astype("float32")
        self.forecast_hours = number_days_forecast * 24
        self.number_of_sequences = len(self.df) // self.forecast_hours

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        self.column_indices = {name: i for i, name in enumerate(self.df.columns)}
        self.feature_columns = [
            col for col in self.df.columns if col not in self.label_columns
        ]

    def __len__(self):
        return self.number_of_sequences

    def __getitem__(self, index):
        inputs = self.df[self.feature_columns].values
        if self.label_columns is not None:
            labels = self.df[[name for name in self.label_columns]].values
        else:
            labels = self.df["electricity"].values
        return inputs.reshape(self.number_of_sequences, self.forecast_hours, -1)[
            index
        ], labels.reshape(self.number_of_sequences, self.forecast_hours, -1)[index]

    def __repr__(self):
        return "\n".join(
            [
                f"Number of sequences: {self.number_of_sequences}",
                f"Label column name(s): {self.label_columns}",
            ]
        )

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, "_example", None)
        if result is None:
            # No example batch was found, so get a random one from the dataset
            random_idx = np.random.randint(0, self.number_of_sequences)
            result = self[random_idx]
            # And cache it for next time
            self._example = result
        return result
