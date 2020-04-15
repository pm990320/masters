import numpy as np
from numpy.lib.stride_tricks import as_strided as stride
import pandas as pd


def cross_validation_split(df, frac_cv=0.3):
    """
    Split a dataframe, containing a time-series, into a training and cross-validation set,
    according to the given split frac_cv

    :param frac_cv: The % of the dataframe to use as the Cross-Validation set.
    :return: (train_df, cv_df)
    """
    split_iloc = len(df) - int(np.floor(len(df) * frac_cv))
    return df.iloc[:split_iloc], df.iloc[split_iloc:]


def remove_anomaly_labels(df):
    new_cols = []
    for col in df.columns:
        if "anomaly" not in col:
            new_cols.append(col)
    return df[new_cols]


def get_anomaly_cols(df):
    new_cols = []
    for col in df.columns:
        if "anomaly_" in col:
            new_cols.append(col)
    return df[new_cols]


def contaminate(df, contamination_factor=0.05, return_contaminated_indices=False):
    """
    Contaminate a certiain % of the dataframe with anomalous random points, but do not label them as anomalous.

    This can be used to show an algorithm's robustness to noise in the input data.
    :param df: The DataFrame to contaminated.
    :param contamination_factor: percentage (0-1) of the dataset to contaminate. Generally <5%.
    :return: a new, contaminated DataFrame.
    """
    num_indices_to_contaminate = int(np.floor(len(df) * contamination_factor))
    if num_indices_to_contaminate < 2:
        if return_contaminated_indices:
            return df, []
        else:
            return df

    start_index = np.random.randint(1, len(df) - num_indices_to_contaminate)
    contaminated_indices = np.arange(
        start_index, start_index + num_indices_to_contaminate
    )

    contamination_coefficients = np.random.random(
        size=(contaminated_indices.shape[0], len(remove_anomaly_labels(df).columns))
    )  # [0, 1)
    contamination_coefficients += 0.5  # [0.5, 1.5)
    sine_wave = np.sin(np.linspace(0, np.pi, num=num_indices_to_contaminate))  # [-1, 1]
    sine_wave = (sine_wave / 2) + 1.5  # [1, 2]
    sine_wave = np.repeat(sine_wave, contamination_coefficients.shape[1]).reshape(
        *contamination_coefficients.shape
    )
    contamination_coefficients *= sine_wave

    contaminated_df = df.copy()
    noanomaly_cdf = remove_anomaly_labels(contaminated_df).copy(deep=True)
    noanomaly_cdf.iloc[contaminated_indices] = (
        noanomaly_cdf.shift(1).fillna(1).iloc[contaminated_indices]
        * contamination_coefficients
    )
    contaminated_df[list(noanomaly_cdf.columns)] = noanomaly_cdf

    if return_contaminated_indices:
        return contaminated_df, contaminated_indices
    else:
        return contaminated_df


def extract_anomaly_labels_to_anomaly_column(df):
    anomaly_cols = []
    for col in df.columns:
        if "anomaly" in col:
            anomaly_cols.append(col)

    anomaly_df = df[anomaly_cols]
    anomaly_col = np.where(anomaly_df.any(axis=1), 1, 0)

    new_df = df.copy()
    new_df["anomaly"] = anomaly_col

    return new_df


def roll(df, w, **kwargs):
    """
    Taken from https://stackoverflow.com/a/38879051

    Utility function to create rolling windows over a dataframe,
    with all columns in that dataframe.

    Intended use - e.g. roll(df, 3).apply(lambda x: x[2])
    """
    v = df.values
    d0, d1 = v.shape
    s0, s1 = v.strides

    a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))

    rolled_df = pd.concat(
        {
            row: pd.DataFrame(values, columns=df.columns)
            for row, values in zip(df.index, a)
        }
    )

    return rolled_df.groupby(level=0, **kwargs)


def upsample(df, upsample_param):
    upsampled_df = df.copy()
    for column in upsampled_df.columns:
        if column.startswith("anomaly_"):
            upsampled_df[column] = upsampled_df[column].astype(int)

    if isinstance(upsampled_df.index, pd.DatetimeIndex):
        upsampled_df = upsampled_df.resample(upsample_param).max()
    else:
        upsampled_df = upsampled_df.set_index(
            pd.Index(
                list(range(1, upsample_param * len(upsampled_df) + 1, upsample_param))
            )
        )
        upsampled_df = upsampled_df.reindex(list(range(1, max(upsampled_df.index) + 1)))

    for column in upsampled_df.columns:
        if "value_" in column:
            upsampled_df[column] = upsampled_df[column].interpolate()
        if "anomaly_" in column:
            upsampled_df[column] = np.ceil(upsampled_df[column].interpolate()).astype(
                bool
            )

    if not isinstance(upsampled_df.index, pd.DatetimeIndex):
        # change to a RangeIndex
        upsampled_df = upsampled_df.reindex(range(1, max(upsampled_df.index) + 1))

    return upsampled_df
