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


def alpha_042(mult, ite, gt, mean_a, plus, std_a, csrank, div, obv_a, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    mult(
        ite(
            gt(mean_2(close),plus(mean_8(close),std_8(close))),
            neg(const_1),
            const_1),
        csrank(div(volume,obv_20()))
        )
    Args: pd.DataFrame
        Data: requires "Adj Close" and "Volume"
    """
    try:
        return mult(
            ite(
                gt(
                    mean_a(data, 2, column="Close")[6:],
                    plus(mean_a(data, 8, column="Close"), std_a(data, 8, column="Close")),
                ),
                -1,
                1,
            )[13:],
            csrank(div(data["Volume"], obv_a(data, 20).dropna())),
        )
    except:
        raise Exception("Operations returned invalid data type, or were not possible")
