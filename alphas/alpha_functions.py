"""
Defines all the alpha functions

Normal alphas will use the following naming comvention:
<p> `alpha_number`, whereas customized alphas that deviate
    from the predefined class variables will have the `custom`
    prefix (e.g. `custom_alpha_num`)
"""
# Third party import
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def alpha_01(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Alpha function
    """
    ma_pair = (20, 60)
    # Compute fast_ewma - slow_ewma
    data[f"ewma({str(ma_pair[0])})"] = data["adj_close"].ewm(span=ma_pair[0]).mean()
    data[f"ewma({str(ma_pair[1])})"] = data["adj_close"].ewm(span=ma_pair[1]).mean()
    data[f"ewma({str(ma_pair[0])}_{str(ma_pair[1])})"] = (
        data[f"ewma({str(ma_pair[0])})"] - data[f"ewma({str(ma_pair[1])})"]
    )
    # Get raw alpha signal
    raw_signal = data[f"ewma({str(ma_pair[0])}_{str(ma_pair[1])})"].rename(ticker)
    # Drop signals on untradeable days
    drop_signal_indices = data["actively_traded"].where(data["actively_traded"] == False).dropna().index
    raw_signal.loc[drop_signal_indices] = 0
    return raw_signal


def alpha__011(data: pd.DataFrame, ticker=str):
    """
    div{rank[minus(mvwap(5), close)],rank(plus{mvwap(5), close})}

    Data: adjusted close column of given dataframe

    rolling window on mwvwap: 5 days
    """

    def MVWAP(data: pd.DataFrame, ticker=str):
        data["Price"] = (data["High"] + data["Low"] + data["Close"]) / 3
        data["Cumulative_Price_Volume"] = (data["Price"] * data["Volume"]).cumsum()
        data["Cumulative_Volume"] = data["Volume"].cumsum()
        data["VWAP"] = data["Cumulative_Price_Volume"] / data["Cumulative_Volume"]
        data["MVWAP"] = data["VWAP"].rolling(5).mean()
        data = data.dropna()

        return data

    def rank(x):

        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        elif not isinstance(x, (pd.Series, int, float)):
            raise TypeError(type(x))

        return np.argsort(x) + 1

    data = MVWAP(data)
    data["Adj Close"] = data["Adj Close"].astype(float)
    data["MVWAP"] = data["MVWAP"].astype(float)
    data["Sum"] = data["MVWAP"] + data["Adj Close"]
    data["Difference"] = data["MVWAP"] - data["Adj Close"]
    rank_difference = rank(data["Difference"])
    rank_sum = rank(data["Sum"])
    raw_signal = rank_difference.div(rank_sum)

    return raw_signal


def alpha_012(data: pd.DataFrame, ticker=str):
    """
    skew - inv{expmn[bscore(54), 15], 72}

    data: adjusted close column
    """

    def bscore(data: pd.Dataframe, b: int):

        mean = data["Adj Close"].iloc[-b:].mean()
        std = data["Adj Close"].iloc[-b:].std()
        data = data.dropna()
        bollinger_score = ((data["Adj Close"][-b:] - mean) / std).dropna()
        return bollinger_score

    def expmn(data: pd.Dataframe, b: int):
        EWM = ((data.ewm(span=b).mean())).dropna()
        return EWM

    def skew(data: pd.Dataframe, b: int):
        return data[-b:].skew()

    bollinger_score = bscore(data, 54)
    EWM = expmn(bollinger_score, 15)
    skew_data = skew(EWM, 72)
    raw_signal = skew_data * -1

    return raw_signal


def alpha_013(data: pd.DataFrame, ticker=str):
    """
    tsmax - inv{cor[tsrank(volume, 5),tsrank(high, 5), 5], 3}
    """

    def ts_rank(x, d):
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        elif not isinstance(x, pd.Series):
            raise TypeError(type(x))
        return x.rolling(d).apply(lambda arr: (np.argsort(np.argsort(arr)) + 1)[d - 1], raw=True)

    def correlation(x, y, d):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x = pd.Series(x)
            y = pd.Series(y)
        elif not isinstance(x, pd.Series) and isinstance(y, pd.Series):
            raise TypeError(type(x))
        return x.rolling(d).corr(y)

    def ts_max(x, d):
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        elif not isinstance(x, pd.Series):
            raise TypeError(type(x))
        return x.iloc[-d:].max()

    volume = data["Volume"]
    high = data["High"]
    rank_volume = (ts_rank(volume, 5)).dropna()
    rank_high = (ts_rank(high, 5)).dropna()
    correlation = (correlation(rank_volume, rank_high, 5)).dropna()
    max = ts_max(correlation, 3)
    raw_signal = max * -1
    return raw_signal


def alpha_014(data: pd.DataFrame, ticker=str):
    """
    plus{tsrank - inv[tsmaxhclose, 6i, 5],tsargmin(tsargmin{bscore(24), 21}, 9)}

    Data Required: Adjusted close

    ****Please note, the original alpha does not work. Code was reworked to pass
    in the minimum of three values into tsargmin(x,3), rather than tsargmin(x,9).
    """

    def bscore(data: pd.Dataframe, b: int):
        mean = data["Adj Close"].iloc[-b:].mean()
        std = data["Adj Close"].iloc[-b:].std()
        data = data.dropna()
        bollinger_score = ((data["Adj Close"][-b:] - mean) / std).dropna()
        return bollinger_score

    def ts_argmin(x, d):
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        elif not isinstance(x, pd.Series):
            raise TypeError(type(x))
        return x.rolling(d).apply(lambda arr: arr.argmax(), raw=True)

    def ts_argmin_2(x, d):
        """augmented tsargmin function to find the timeseries's minimum index looking
        back at 3 values. So tsargmin(x,3) rathern than tsargmin(x,9)"""
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        elif not isinstance(x, pd.Series):
            raise TypeError(type(x))
        return 1 + np.argmin(x[-d:])

    def ts_max(x, d):
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        elif not isinstance(x, pd.Series):
            raise TypeError(type(x))
        return x.rolling(d).max()

    def tsrank_a(data: pd.DataFrame, a: int) -> pd.DataFrame:
        try:
            return stats.rankdata(data[-a:], method="average", nan_policy="omit")[-1]
        except:
            raise Exception(f"Data is not long enough for lookback period {a}")

    def plus(data_a: pd.DataFrame, data_b: pd.DataFrame) -> pd.DataFrame:
        try:
            return data_a + data_b
        except:
            raise Exception("The summands are not the same datatypes and cannot be summed")

    close = data["Adj Close"]
    bollinger_score = bscore(data, 24)
    tsargmin_1 = ts_argmin(bollinger_score, 21)
    tsargmin_2 = ts_argmin_2(tsargmin_1, 3)
    print(tsargmin_2)
    tsmax = (ts_max(close, 6)).dropna()
    inv_rank = tsrank_a(tsmax, 5) * -1
    raw_signal = plus(inv_rank, tsargmin_2)
    return raw_signal


def alpha_015(data: pd.DataFrame, ticker=str):
    def minus(data_a: pd.DataFrame, data_b: pd.DataFrame) -> pd.DataFrame:
        """Difference of two dataframes"""
        try:
            return data_a - data_b
        except:
            raise Exception("The summands are not the same datatypes and cannot be subtracted")

    def div(data_a: pd.Series, data_b: pd.Series) -> pd.DataFrame:
        """Quotient of two columns"""
        try:
            return data_a / data_b
        except:
            raise Exception("The summands are not the same datatypes and cannot be subtracted")

    close = data["Close"].iloc[-1]
    open = data["Open"].iloc[-1]
    high = data["High"].iloc[-1]
    low = data["Open"].iloc[-1]
    difference_1 = open - close
    difference_2 = high - low
    raw_signal = div(difference_1 / difference_2)
    return raw_signal
