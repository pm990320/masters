import os
import json

import pandas as pd

NYC_TAXI_FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "nab/data/realKnownCause/nyc_taxi.csv",
)
NAB_LABELS_FILE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "nab/labels/combined_windows.json"
)


def nab_nyc_taxi():
    # obtain the anomalous windows
    with open(NAB_LABELS_FILE_PATH, "r") as labels_file:
        all_series_windows = json.load(labels_file)

    df = pd.read_csv(NYC_TAXI_FILE_PATH)
    df.set_index("timestamp", inplace=True)

    anomaly_windows = all_series_windows[f"realKnownCause/nyc_taxi.csv"]
    df["anomaly"] = False  # assume all points are Normal
    for window_start, window_end in anomaly_windows:
        df.loc[window_start:window_end, "anomaly"] = True

    return df
