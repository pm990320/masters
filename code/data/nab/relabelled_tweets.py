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
RELABELLED_TWEET_FILE_PATH_FORMAT = os.path.join(
    os.path.dirname(__file__), "relabelled", "{stock}-labeled.csv"
)


def nab_tweets_relabelled():
    result = None

    for stock in STOCK_NAMES:
        filename = RELABELLED_TWEET_FILE_PATH_FORMAT.format(stock=stock)
        df = pd.read_csv(filename)
        df["timestamp"] = df["timestamp"].apply(lambda x: x[:-5])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        df["anomaly"] = df["label"].astype(bool)
        del df["label"]
        del df["filename"]

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
