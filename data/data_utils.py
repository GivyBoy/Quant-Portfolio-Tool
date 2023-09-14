"""
Utility class for the Data module
"""
from __future__ import annotations

import datetime as dt
import pickle

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


def dict_to_df(data: dict) -> pd.DataFrame:
    """
    Converts a multi-dimensional dictionary (e.g., what get_ohlc_data() returns) to a
    multi-index pd.DataFrame.

    Args:
        data (dict): general multi-dimensional dict.

    Returns:
        pd.DataFrame: newly converted multi-dimensional object.
    """
    return pd.concat(data, axis=0, keys=data.keys())


def cache(data, path: str):
    """
    Flexible function to pickle/cache any data structure to a given path.

    Args:
        data (any): data structure to cache.
        path (str): cache file path. Must include file extension within path.
    """
    print(f"Writing cache to {path}")
    try:
        with open(path, "wb") as writer:
            pickle.dump(data, writer, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as err:
        print(err)


def load_cache(path: str):
    """
    Flexible function to read pickle/cache and return the data structure from a given
    path.

    Args:
        path (str): cache file path. Must include file extension within path.
    """
    print(f"Loading cache from {path}")
    try:
        with open(path, "rb") as reader:
            data = pickle.load(reader)
        return data
    except Exception as err:
        print(err)


def get_sp500_tickers() -> list:
    """
    Reads a list of S&P 500 tickers from Wikipedia, then returns a list of them.

    Returns:
        list: list of S&P 500 tickers.
    """
    return list(pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0].Symbol)


def extend_ohlc_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-processes and adds relevant statistics to existing OHLC pd.DataFrame.

    Args:
        data (pd.DataFrame): OHLC data.

    Returns:
        pd.DataFrame: pre-processed and extended OHLC data.
    """
    # Fill missing data by first forward filling,
    # such that [] [] [] a b c [] [] [] becomes [] [] [] a b c c c c
    data.fillna(method="ffill", inplace=True)
    # Fill missing data by then backwards filling,
    # such that [] [] [] a b c c c c becomes a a a a b c c c c
    data.fillna(method="bfill", inplace=True)

    # Compute return across entire OHLC specturm
    for stat in ["open", "high", "low", "close", "adj_close"]:
        # Log returns
        data[f"log_{stat}_returns"] = np.log(data[stat] / data[stat].shift(1)).dropna()
        # Arithmetic returns
        data[f"{stat}_returns"] = data[stat].pct_change().dropna()

    # Indicate if instrument is actively traded - rough measure
    data["actively_traded"] = (
        (data["adj_close_returns"] != np.inf) | (data["adj_close_returns"] != -1.0) | (data["adj_close_returns"] != 0)
    )

    # Replace np.inf with np.nan
    for stat in ["open", "high", "low", "close", "adj_close"]:
        data[f"{stat}_returns"] = data[f"{stat}_returns"].replace(np.inf, np.nan)
    return data


def get_ohlc_data(ticker_list: list) -> dict:
    """
    Gets OHLC data and computes returns data for each instrument in "ticker_list".
    Stores each pd.DataFrame into a master universe_ohlc_data dictionary whose key-value
    pair = ticker : data.

    Args:
        ticker_list (list): list of equity tickers.

    Returns:
        dict: master dictionary storing each stock's OHLC + returns data.
    """
    # Initialize universe dict
    universe_ohlc_data = {}

    for ticker in tqdm(ticker_list):
        try:
            data = yf.download(ticker, start="1980-01-01", end=dt.date.today())
            data = data.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            # Get ancillary statistics
            data = extend_ohlc_statistics(data)

            # Convert index to datetime.date objects
            data.index = data.index.date

            # Store instrument's data in universe dict
            universe_ohlc_data[ticker] = data

        except Exception as err:
            print(err)
    return universe_ohlc_data


def cache_cross_universe_statistics(ohlc_data: dict, path: str) -> None:
    """
    Caches all statistics in a cross-universe manner. This creates a database structure
    that offers access to a single statistic across an entire stock universe vs.
    a required traversal through tickers anytime the researcher wants to grab a
    specific statistic.

    Args:
        ohlc_data (dict): dictionary whose key-value pair = ticker : data.
        path (str): directory path to cache cross-universe statistics.
    """
    # Get stock statistics list
    stat_list = ohlc_data[next(iter(ohlc_data))].columns

    # Cache cross-universe statistics
    for stat in stat_list:
        # Initialize cross-universe statistics dictionary and statistic-specific
        # cache path
        tmp_stat_dict = {}
        tmp_path = path + rf"/{stat}.pickle"

        # Get statisitc for each instrument
        for ticker, data in ohlc_data.items():
            tmp_stat_dict[ticker] = data[stat]

        cache(data=tmp_stat_dict, path=tmp_path)
