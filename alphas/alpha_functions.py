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

# ALPHAS

def alpha_021(data): 
    try:
        return inv(expmn(div(kurt(returns(data, 1, "Close"), 10), std(low(data), 10)), 5))
    except:
        print("alpha_021 has an error")

def alpha_022(data): # HAS ADV
    return inv(mean(cov(ADX(data, 7), std(data["Close"], 25), 7), 7))

def alpha_023(data): # HAS ADV
    return inv(minus(rank(minus(mvwap(data, 5), tsmin(mvwap(data, 5), 10))), rank(cor(mvwap(data,5), ADX(data,8), 10))))

def alpha_024(data):
    return inv(mult(cov(high(data), returns(data, 30, "Close"), 41), csscale(volume(data))))

def alpha_025(data): # HAS ADV
    return inv(sum_(mean(cov(ADX(data,67), ADX(data,90), 32), 14), 56))


def alpha_026(data): # BECOMING DOGSHIT
    return cov(inv(minus(abs_(mvwap(data,90)),rank(low(data)))), inv(np.array(delta(rank(close(data)), 51))), 55) 

def alpha_027(data):
    return delta(div(low(data), close(data)), 26)

def alpha_028(data):
    return expmn(inv(mult(cov(high(data),returns(data, 3, "Close"), 5), std(std(data["Close"], 25), 5))),5)

def alpha_029(data):
    return alpha_028(data) # GET FUCKING PRANKED

def alpha_030(data):
    return minus(np.array(rank(returns(data, 10, "Close"))), np.array(rank(std(high(data), 10)))) 
# End
