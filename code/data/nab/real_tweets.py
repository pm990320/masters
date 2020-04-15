import os
import json

import pandas as pd

STOCK_NAMES = [
    "AAPL",
    "AMZN",
    "CRM",
    "CVS",
    "FB",
    "GOOG",
    "IBM",
    "KO",
    "PFE",
    "UPS",
]
NAB_TWEET_FILE_PATH_FORMAT = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "nab/data/realTweets/Twitter_volume_{stock}.csv",
)
NAB_LABELS_FILE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "nab/labels/combined_labels.json"
)
NAB_WINDOWS_FILE_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "nab/labels/combined_windows.json"
)


def nab_multivariate_tweet_volume(labels="windows"):
    # obtain the anomalous windows
    with open(
        NAB_WINDOWS_FILE_PATH if labels == "windows" else NAB_LABELS_FILE_PATH, "r"
    ) as labels_file:
        label_json = json.load(labels_file)

    result = None

    for stock in STOCK_NAMES:
        filename = NAB_TWEET_FILE_PATH_FORMAT.format(stock=stock)
        df = pd.read_csv(filename)
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index)

        df["anomaly"] = False  # assume all points are Normal

        seires_labels = label_json[f"realTweets/Twitter_volume_{stock}.csv"]
        if labels == "windows":
            for window_start, window_end in seires_labels:
                df.loc[window_start:window_end, "anomaly"] = True
        else:
            for label in seires_labels:
                df.loc[label, "anomaly"] = True

        if result is None:
            result = df
            result[f"value_{stock}"] = result["value"]
            result[f"anomaly_{stock}"] = result["anomaly"]
        else:
            result = result.join(df, how="outer", rsuffix=f"_{stock}")

    del result["value"]
    del result["anomaly"]
    result.fillna(0, inplace=True)

    return result
