from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from scipy import stats


def mean_a(data: pd.DataFrame, a: int, column: str = "Adj Close") -> pd.DataFrame:
    """Rolling Mean; with period 'a'"""
    try:
        return data["Adj Close"].rolling(window=a).mean().dropna()
    except:
        raise Exception(f"Column {column} is not in {data}")


def plus(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
) -> pd.DataFrame:
    """Sum of two dataframes"""
    try:
        return data_a + data_b
    except:
        raise Exception("The summands are not the same datatypes and cannot be summed")


def minus(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
) -> pd.DataFrame:
    """Difference of two dataframes"""
    try:
        return data_a - data_b
    except:
        raise Exception("The summands are not the same datatypes and cannot be subtracted")


def neg(data: pd.DataFrame) -> pd.DataFrame:
    """Returns negative of input"""
    try:
        return -data
    except:
        raise Exception("The data cannot be made negative")


def std_a(data: pd.DataFrame, a: int, column: str = "Adj Close") -> pd.DataFrame:
    """Rolling Standard Deviation; with period 'a'"""
    try:
        return data["Adj Close"].rolling(window=a).mean().dropna()
    except:
        raise Exception(f"Column {column} is not in {data}")


def mvwap_a(data: pd.DataFrame, a: int) -> pd.DataFrame:
    """Moving Volume Weighted Average Price; with period 'a'"""
    try:
        return ((data["High"] + data["Low"] + data["Close"]) / 3 * data["Volume"]).cumsum() / data[
            "Volume"
        ].cumsum().rolling(window=a).mean().dropna()
    except:
        raise Exception(
            "All necessary columns are not present in the data; the function requires 'High', 'Low', 'Close', "
            "and 'Volume)"
        )


def obv_a(data: pd.DataFrame, a: int) -> pd.DataFrame:
    """On Balance Volume; with period 'a'"""
    try:
        return (
            np.sign(((data["Adj Close"] / data["Adj Close"].shift(1) - 1) * data["Volume"]).dropna())
            .rolling(window=a)
            .sum()
            .dropna()
        )
    except:
        raise Exception(
            "All necessary columns are not present in the data; the function requires 'Adj Close' and 'Volume'"
        )


def tsrank_a(data: pd.DataFrame, a: int) -> pd.DataFrame:
    """Time Series Rank of last element in data; with lookback 'a'"""
    try:
        return stats.rankdata(data[-a:], method="average", nan_policy="omit")[-1]
    except:
        raise Exception(f"Data is not long enough for lookback period {a}")


def kentau_a(data_a: pd.DataFrame, data_b: pd.DataFrame, a: int) -> pd.DataFrame:
    """Kendall-Tau Correlation of two datasets; with lookback 'a'"""
    try:
        return scipy.stats.kendalltau(data_a[-a:], data_b[-a:])[0]
    except:
        raise Exception("data is not compatible for kendall-tau correlation")


def gt(data_a: pd.DataFrame, data_b: pd.DataFrame) -> pd.DataFrame:
    """Return Boolean Datarame of "a" > "b" """
    try:
        return data_a > data_b
    except:
        raise Exception("data cannot be compared")


def ite(x, y: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """If x, then y; else z"""
    try:
        x.fillna(0).astype(int) * y + (~x.astype(bool)).fillna(0).astype(int) * z
    except:
        raise Exception("Data cannot be substituted, boolean may not be same dimensions as dataframes")
