"""
Utility functions for the `alpha_builder.py` file
"""
from __future__ import annotations

import pandas as pd


def quantiles(signal: pd.Series, q=0.1):
    """
    Isolates values in the bottom and top quantiles of a signal.

    Parameters:
        signal (pd.Series): Series of alpha signals.
        q (float, optional): Quantile threshold (default: 0.1).

    Returns:
        pd.Series: Signal with values outside the quantiles set to NaN.
    """

    lower_quantile = signal.quantile(q=q)
    upper_quantile = signal.quantile(q=1 - q)

    quantile_signal = signal.where((signal < lower_quantile) | (signal > upper_quantile)).dropna()

    return quantile_signal
