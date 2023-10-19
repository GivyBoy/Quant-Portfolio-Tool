"""
Defines all the alpha functions

Normal alphas will use the following naming comvention:
<p> `alpha_number`, whereas customized alphas that deviate
    from the predefined class variables will have the `custom`
    prefix (e.g. `custom_alpha_num`)
"""
# Third party import
from __future__ import annotations

import pandas as pd  # used for data analysis, cleaning and manipulation
from alpha_utils import *  # noqa 401, 403


def get_stock_data(tikr, begin="2010-01-01", stop="2022-01-01"):
    return yf.download(str((tikr)), start=begin, end=stop)


def alpha_059(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
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


def alpha_001(data: pd.DataFrame) -> pd.Series:
    """
    expmn − inv{mult[rank(obv(15)),tsmax(volume, 12)], 10}
    """

    var_x = neg([ts_rank(obv_a(15)) * tsmax_a(data["Volume"], 12)])

    return ewma_a(var_x, 10)

    # volume = data["Volume"]
    # yash added tsmax, inv, mult; used * instead of mult (talk to anthony)
    # added obv_a but need to ask anthony because didn't realize there already was one
    # ask about ewma / expmn

    pass


def alpha_002(data: pd.DataFrame) -> pd.Series:
    """
    mult − inv{cov[high, low, 12],std(volatility, 20)}
    """
    neg_covariance = neg(covariance[data("High"), data("Low"), 12])
    neg_std_close = std(std(data("Close"), 25), 20)

    return neg_covariance * neg_std_close

    # ask anthony how to do the VOLATILITY - rolling 25 day std - std(data("Close")) (important for alpha 4)


def alpha_003(data: pd.DataFrame) -> pd.Series:
    """
    abs − inv{csscale[cov(bscore(97), adv(63), 86)]}
    """
    neg_cs_scale = neg(csscale[covariance(bscore(97), adx(63), 86)])
    return abs(neg_cs_scale)

    # issue finding cs scale in documentation
    # issue finding bscore in documentation
    # ask anthony about the const_100 with adx + confirm it is right util


def alpha_004(data: pd.DataFrame) -> pd.Series:
    """
    abs − inv{mult[stdhmvwap(49), 35i, cor(volatility,returns(56), 24)]}
    """

    std_mvwap_a = std(mvwap_a(49, 35))
    cor_vol_returns = correlation(std(data("Close"), 25), returns(56), 24)

    return abs(neg(std_mvwap_a * cor_vol_returns))

    # figure out how to do returns/ which returns in documentation
    # check on volatility

    pass


def alpha_005(data: pd.DataFrame) -> pd.Series:
    """
    tsrank − inv{expmn[bscore(25), 40], 70}
    """
    neg_expmn_bscore = neg(expmn[bscore(25), 40], 70)
    return tsrank_a(neg_expmn_bscore)

    # ask anthony if expmn is ewma
    # have problem finding bscore in documentation

    pass


# data = get_stock_data("^GSPC")
# x = alpha_001(data)

# print(data["Volatility"])
