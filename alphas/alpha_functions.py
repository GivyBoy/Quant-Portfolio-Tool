"""
Defines all the alpha functions

Normal alphas will use the following naming comvention:
<p> `alpha_number`, whereas customized alphas that deviate
    from the predefined class variables will have the `custom`
    prefix (e.g. `custom_alpha_num`)
"""
# Third party import
from __future__ import annotations

import pandas as pd
from quant_portfolio_tool.alphas.alpha_utils import *


def alpha_001(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
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


def alpha_002(data: pd.DataFrame) -> pd.Series:
    return -1 * correlation(
        rank(delta(log(data["volume"]), 2)), rank(((data["close"] - data["open"]) / data["open"])), 6
    )


def alpha_003(data: pd.DataFrame) -> pd.Series:
    return -1 * correlation(rank(data["close"]), rank(data["volume"]), 10)


def alpha_004(data: pd.DataFrame) -> pd.Series:
    return -1 * ts_rank(rank(data["low"]), 9)


def alpha_005(data: pd.DataFrame) -> pd.Series:
    return -1 * ts_rank(rank(data["low"]), 9)


def alpha_006(data: pd.DataFrame) -> pd.Series:
    return rank((open - (sum(data["vwap"], 10) / 10))) * (-1 * abs(rank((data["close"] - data["vwap"]))))


def alpha_007(data: pd.DataFrame) -> pd.Series:
    return -1 * correlation(data["open"], data["volume"], 10)


def alpha_008(data: pd.DataFrame) -> pd.Series:
    adv20 = mean(data["volume"], 20)
    return np.where(
        adv20 < data["volume"], -1 * ts_rank(abs(delta(data["close"], 7)), 60) * sign(delta(data["close"], 7)), -1
    )


def alpha_009(data: pd.DataFrame) -> pd.Series:
    return -1 * rank(
        (
            (sum(data["open"], 5) * sum(data["close_returns"], 5))
            - delay((sum(data["open"], 5) * sum(data["close_returns"], 5)), 10)
        )
    )


def alpha_010(data: pd.DataFrame) -> pd.Series:
    return np.where(
        ts_min(delta(data["close"], 1), 5) > 0,
        delta(data["close"], 1),
        np.where(ts_max(delta(data["close"], 1), 5) < 0, delta(data["close"], 1), -1 * delta(data["close"], 1)),
    )
