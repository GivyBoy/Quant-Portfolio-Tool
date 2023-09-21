from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from scipy import stats


def mean_a(data: pd.DataFrame, a: int, column: str = "Adj Close") -> pd.DataFrame:
    """Rolling Mean; with period 'a'"""
    try:
        return data[column].rolling(window=a).mean().dropna()
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


def mult(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
) -> pd.DataFrame:
    """Difference of two dataframes"""
    try:
        return data_a * data_b
    except:
        raise Exception("The elements are not compatible datatypes and cannot be multiplied")


def div(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
) -> pd.DataFrame:
    """Difference of two dataframes"""
    try:
        return data_a / data_b
    except:
        raise Exception("The elements are not compatible datatypes and cannot be divided")


def neg(data: pd.DataFrame) -> pd.DataFrame:
    """Returns negative of input"""
    try:
        return -data
    except:
        raise Exception("The data cannot be made negative")


def std_a(data: pd.DataFrame, a: int, column: str = "Adj Close") -> pd.DataFrame:
    """Rolling Standard Deviation; with period 'a'"""
    try:
        return data[column].rolling(window=a).mean().dropna()
    except:
        raise Exception(f"Column {column} is not in {data}")


def ewma_a(
    data: pd.DataFrame,
    a: int,
) -> pd.DataFrame:
    """Exponentially weighted moving average over meaurement period 'a'"""
    try:
        return data["Adj Close"].ewm(span=a, adjust=False).mean()
    except:
        raise Exception(f"Column 'Adj Close' is not in {data}")


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


def delta_a(data: pd.DataFrame, a: int) -> pd.DataFrame:
    """Change in data vector over last 'a' days"""
    try:
        return data.diff(periods=a).dropna()
    except:
        raise Exception("Data does not have enough periods to have lookback 'a'")


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


def csrank(data: pd.DataFrame, x: str) -> pd.DataFrame:
    """Cross sectional rank of column 'x' in data"""
    try:
        return scipy.stats.rankdata(x, method="average", nan_policy="omit")
    except:
        raise Exception(f"Data does not have column {x}")


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
    """If x, then y; else z, where x is a boolean matrix"""
    try:
        x.fillna(0).astype(int) * y + (~x.astype(bool)).fillna(0).astype(int) * z
    except:
        raise Exception("Data cannot be substituted, boolean might not be same dimensions as dataframes")
