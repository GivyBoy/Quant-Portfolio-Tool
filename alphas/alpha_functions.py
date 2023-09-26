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

import pandas_ta as ta

from alpha_utils import *

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

def alpha_006(data: pd.DataFrame) -> pd.Series:
    """
    div{inv[delta(bscore(90),14)],tsrank(delta{bscore(13),36},38)}
    """
    neg_bscore_delta = neg(delta(bscore(data['Adj Close'], 90), 14))
    tsrank_bscore_delta = ts_rank(delta(bscore(data['Adj Close'], 13), 36), 38)

    return neg_bscore_delta/tsrank_bscore_delta

def alpha_007(data: pd.DataFrame) -> pd.DataFrame:
    """
    inv{mult(rank[tsrank{close,10}], rank[div{close,open}])}
    """
    rank_tsrank_close = ts_rank(data['Adj Close'], 10).rank()
    rank_close_open_div = (data['Adj Close'] / data['Open']).rank()

    return neg(rank_tsrank_close * rank_close_open_div)

def alpha_008(data: pd.DataFrame) -> pd.DataFrame:
    """
    tsmin(inv[mult{cov(high,returns(30),41),csscale(volume)}], 13)
    """
    cov = covariance(data['High'], returns(data['Adj Close'], 30), 41)
    csscale_vol = csscale(data['Volume']) 

    return ts_min(neg(cov * csscale_vol), 13)

def alpha_009(data: pd.DataFrame) -> pd.DataFrame:
    """
    inv(sum{mean[cov(adv(67),volatility,32),14],56})
    """
    cov = covariance(ta.adx(data['High'], data['Low'], data['Adj Close'], length=67)['ADX_67'], std_a(data,25), 32)

    return neg(sum(mean(cov, 14), 56))




def alpha_010(data: pd.DataFrame) -> pd.DataFrame:
    """
    inv{rank(cov[rank(close),rank(volume),3])}
    """ 
    cov_ranked = covariance(data['Adj Close'].rank(), data['Volume'].rank(), 3) 

    return neg(cov_ranked.rank())
