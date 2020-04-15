import os
import json

import pandas as pd


A1_FILE_PATH_FORMAT = os.path.join(
    os.path.dirname(__file__), "A1Benchmark/real_{num}.csv"
)

SERIES_TO_EXCLUDE = set(
    [
        54,
        62,
        # from training set rationalisation
        1,
        2,
        4,
        6,
        10,
        11,
        16,
        21,
        22,
        25,
        31,
        32,
        33,
        35,
        37,
        42,
        45,
        50,
        58,
        59,
        63,
        64,
        65,
        66,
        67,
    ]
)


def yahoo_a1_benchmark(clean_series=True):
    result = None

    for num in range(1, 68):
        if clean_series and num in SERIES_TO_EXCLUDE:
            continue

        filename = A1_FILE_PATH_FORMAT.format(num=num)
        df = pd.read_csv(filename)
        df.set_index("timestamp", inplace=True)
        df["anomaly"] = df["is_anomaly"].astype(bool)
        del df["is_anomaly"]

        if result is None:
            result = df
            result[f"value_{num}"] = result["value"]
            result[f"anomaly_{num}"] = result["anomaly"]
        else:
            result = result.join(df, how="outer", rsuffix=f"_{num}")

    del result["value"]
    del result["anomaly"]

    result.fillna(0, inplace=True)
    result = result.reindex(range(1, max(result.index) + 1))

    return result
