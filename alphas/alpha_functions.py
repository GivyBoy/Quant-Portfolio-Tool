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

import alpha_utils as utils

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
    neg_bscore_delta = utils.neg(utils.delta(utils.bscore(data['Adj Close'], 90), 14))
    tsrank_bscore_delta = utils.ts_rank(utils.delta(utils.bscore(data['Adj Close'], 13), 36), 38)

    result = neg_bscore_delta/tsrank_bscore_delta
    return result

def alpha_007(data: pd.DataFrame) -> pd.DataFrame:
    """
    inv{mult(rank[tsrank{close,10}], rank[div{close,open}])}
    """
    rank_tsrank_close = utils.ts_rank(data['Adj Close'], 10).rank()
    rank_close_open_div = (data['Adj Close'] / data['Open']).rank()

    result = utils.neg(rank_tsrank_close * rank_close_open_div)
    return result

def alpha_008(data: pd.DataFrame) -> pd.DataFrame:
    """
    tsmin(inv[mult{cov(high,returns(30),41),csscale(volume)}], 13)
    """
    cov = utils.covariance(data['High'], utils.returns(data['Adj Close'], 30), 41)
    csscale_vol = utils.csscale(data['Volume']) 

    result = utils.ts_min(utils.neg(cov * csscale_vol), 13)
    return result

def alpha_009(data: pd.DataFrame) -> pd.DataFrame:
    """
    inv(sum{mean[cov(adv(67),volatility,32),14],56})
    """
    cov = utils.covariance(utils.adv(data, 67), utils.volatility(data), 32)

    result = utils.neg(utils.sum(utils.mean(cov, 14), 56))
    return result

def alpha_010(data: pd.DataFrame) -> pd.DataFrame:
    """
    inv{rank(cov[rank(close),rank(volume),3])}
    """ 
    cov_ranked = utils.covariance(data['Adj Close'].rank(), data['Volume'].rank(), 3) 

    result = utils.neg(cov_ranked.rank())
    return result
