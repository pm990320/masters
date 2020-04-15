import os
import json

import pandas as pd

ARTIFICIAL_SERIES = [
    "art_daily_flatmiddle",
    "art_daily_jumpsup",
    "art_daily_jumpsdown",
]
ARTIFICIAL_FILE_PATH_FORMAT = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "nab/data/artificialWithAnomaly/{series}.csv",
)
NAB_LABELS_FILE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "nab/labels/combined_windows.json"
)


def nab_artificial():
    # obtain the anomalous windows
    with open(NAB_LABELS_FILE_PATH, "r") as labels_file:
        all_series_windows = json.load(labels_file)

    result = None

    for series in ARTIFICIAL_SERIES:
        filename = ARTIFICIAL_FILE_PATH_FORMAT.format(series=series)
        df = pd.read_csv(filename)
        df.set_index("timestamp", inplace=True)

        anomaly_windows = all_series_windows[f"artificialWithAnomaly/{series}.csv"]
        df["anomaly"] = False  # assume all points are Normal
        for window_start, window_end in anomaly_windows:
            df.loc[window_start:window_end, "anomaly"] = True

        if result is None:
            result = df
            result[f"value_{series}"] = result["value"]
            result[f"anomaly_{series}"] = result["anomaly"]
        else:
            result = result.join(df, how="outer", rsuffix=f"_{series}")

    del result["value"]
    del result["anomaly"]
    result.fillna(0, inplace=True)

    return result
