from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from scipy import stats


def abs(x):
    """
    Calculates the absolute values of an array or scalar.
    Parameters:
        x (pd.Series, np.ndarray, float, int): Input data.
    Returns:
        pd.Series, np.ndarray, float, int: Absolute values of the input data.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, (pd.Series, int, float)):
        raise TypeError(type(x))

    return np.abs(x)


def log(x):
    """
    Calculates the natural logarithm of an array or scalar.
    Parameters:
        x (pd.Series, np.ndarray, float, int): Input data.
    Returns:
        pd.Series, np.ndarray, float, int: Natural logarithm of the input data.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, (pd.Series, int, float)):
        raise TypeError(type(x))

    return np.log(x)


def sign(x):
    """
    Calculates the sign of an array or scalar.
    Parameters:
        x (pd.Series, np.ndarray, float, int): Input data.
    Returns:
        pd.Series, np.ndarray, float, int: Signs of the input data (-1, 0, or 1).
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, (pd.Series, int, float)):
        raise TypeError(type(x))

    return np.sign(x)


def rank(x):
    """
    Cross-sectionally ranks alpha signal.
    Parameters:
        x (pd.Series or np.ndarray): Series of alpha signals.
    Returns:
        pd.Series or np.ndarray: Ranked signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, (pd.Series, int, float)):
        raise TypeError(type(x))

    return np.argsort(x) + 1


def delay(x, d):
    """
    Delays the input signal by a specified number of periods.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Number of periods to delay the signal.
    Returns:
        pd.Series, np.ndarray: Delayed signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    return np.roll(x, d)


def correlation(x, y, d):
    """
    Calculates the rolling correlation between two signals over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): First input signal.
        y (pd.Series, np.ndarray): Second input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling correlation between the two signals.
    """

    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x = pd.Series(x)
        y = pd.Series(y)
    elif not isinstance(x, pd.Series) and isinstance(y, pd.Series):
        raise TypeError(type(x))

    return x.rolling(d).corr(y)


def covariance(x, y, d):
    """
    Calculates the rolling covariance between two signals over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): First input signal.
        y (pd.Series, np.ndarray): Second input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling covariance between the two signals.
    """

    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        x = pd.Series(x)
        y = pd.Series(y)
    elif not isinstance(x, pd.Series) and isinstance(y, pd.Series):
        raise TypeError(type(x))

    return x.rolling(d).cov(y)


def scale(x, scalar=1):
    """
    Scales an input signal by a specified factor.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        scalar (float): Scaling factor (default: 1).
    Returns:
        pd.Series, np.ndarray: Scaled signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, (pd.Series, int, float)):
        raise TypeError(type(x))

    return x * scalar / np.abs(x).sum()


def delta(x, d):
    """
    Calculates the difference between a signal and its delayed version.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Number of periods to delay the signal.
    Returns:
        pd.Series, np.ndarray: Signal representing the difference between the original and delayed signals.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    return x - delay(x, d)


def power(x, a):
    """
    Calculates the element-wise power of an input signal raised to a specified exponent.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        a (float): Exponent for the power operation.
    Returns:
        pd.Series, np.ndarray: Signal with elements raised to the specified exponent.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, (pd.Series, int, float)):
        raise TypeError(type(x))

    return np.power(x, a)


def ts_min(x, d):
    """
    Calculates the rolling minimum of a signal over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling minimum of the input signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    return x.rolling(d).min()


def ts_max(x, d):
    """
    Calculates the rolling maximum of a signal over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling maximum of the input signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    return x.rolling(d).max()


def ts_argmax(x, d):
    """
    Calculates the rolling index of the maximum value in a signal over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling index of the maximum value in the input signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    return x.rolling(d).apply(lambda arr: arr.argmax(), raw=True)


def ts_argmin(x, d):
    """
    Calculates the rolling index of the minimum value in a signal over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling index of the minimum value in the input signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    return x.rolling(d).apply(lambda arr: arr.argmax(), raw=True)


def ts_rank(x, d):
    """
    Calculates the rolling rank of values in a signal over a specified window.
    The custom ranking function (arr.argsort().argsort() + 1)[d - 1] calculates
    the rank of each element in the rolling window, where arr represents the elements
    in the rolling window. The .argsort() method sorts the elements in ascending order,
    and .argsort().argsort() effectively returns the rank of each element.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling rank of values in the input signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    return x.rolling(d).apply(lambda arr: (np.argsort(np.argsort(arr)) + 1)[d - 1], raw=True)


def min(x, d):
    """
    Calculates the rolling minimum of a signal over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling minimum of the input signal.
    """

    return ts_min(x, d)


def max(x, d):
    """
    Calculates the rolling maximum of a signal over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling maximum of the input signal.
    """

    return ts_max(x, d)


def sum(x, d):
    """
    Calculates the rolling sum of a signal over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling sum of the input signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    return x.rolling(d).sum()


def product(x, d):
    """
    Calculates the rolling product of a signal over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling product of the input signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    return x.rolling(d).apply(lambda arr: np.prod(arr), raw=True)


def mean(x, d):
    """
    Calculates the rolling standard deviation of a signal over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling standard deviation of the input signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    return x.rolling(d).mean()


def std(x, d):
    """
    Calculates the rolling standard deviation of a signal over a specified window.
    Parameters:
        x (pd.Series, np.ndarray): Input signal.
        d (int): Rolling window size.
    Returns:
        pd.Series: Rolling standard deviation of the input signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    return x.rolling(d).std()


def quantiles(x: pd.Series, q=0.1):
    """
    Isolates values in the bottom and top quantiles of a signal.
    Parameters:
        x (pd.Series): Series of alpha signals.
        q (float, optional): Quantile threshold (default: 0.1).
    Returns:
        pd.Series: Signal with values outside the quantiles set to NaN.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    lower_quantile = x.quantile(q=q)
    upper_quantile = x.quantile(q=1 - q)
    quantile_signal = x.where((x < lower_quantile) | (x > upper_quantile)).dropna()

    return quantile_signal


def quantile_votes(x: pd.Series, q=0.1):
    """
    Isolates values in the bottom and top quantiles of a signal.
    Assigns indicator votes to bottom and top quantiles.
    Parameters:
        x (pd.Series): Series of alpha signals.
        q (float, optional): Quantile threshold (default: 0.1).
    Returns:
        pd.Series: Signal with values outside the quantiles set to NaN.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    lower_quantile = x.quantile(q=q)
    upper_quantile = x.quantile(q=1 - q)

    shorts_index = list(x.where(x < lower_quantile).dropna().index)
    longs_index = list(x.where(x > upper_quantile).dropna().index)

    quantile_votes = pd.Series(index=shorts_index + longs_index)
    quantile_votes.loc[shorts_index] = -1
    quantile_votes.loc[longs_index] = 1

    return quantile_votes


def z_score(x: pd.Series):
    """
    Calculates the z-score of a signal.
    Parameters:
        x (pd.Series): Series of the signal to calculate Z-scores for.
    Returns:
        pd.Series: Z-scores of the input signal.
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    z_score = (x - np.mean(x)) / np.std(x)

    return z_score


def ranked_signal_z_score(x: pd.Series):
    """
    Calculates z-scores for a ranked signal and ensures dollar-neutrality.
    Parameters:
        x (pd.Series): Series of the signal to calculate Z-scores for.
    Returns:
        pd.Series: Z-scores of the ranked signal.
    Raises:
        AssertionError: If the sum of Z-scores is not close to zero (dollar-neutrality check).
    """

    if isinstance(x, np.ndarray):
        x = pd.Series(x)
    elif not isinstance(x, pd.Series):
        raise TypeError(type(x))

    # Rank signals
    ranked_signal = rank(x)

    # Standardize ranked signals
    z_score_signal = z_score(ranked_signal)

    # Check for dollar-neutrality
    assert -1e-10 < z_score_signal.sum() < 1e-10

    # Dollar-neutral alpha model views
    return z_score_signal


def intra_industry_rank(x: pd.Series, industry: pd.Series):
    """
    Computes intra-industry cross-sectional ranks for a signal.
    Parameters:
        x (pd.Series): Series of the signal to calculate intra-industry ranks for.
        industry (pd.Series): Series indicating the industry of each observation.
    Returns:
        pd.Series: Intra-industry ranks of the signal.
    """
    # Create intra-industry ranked signals
    ranked_signal = x.groupby(industry).rank()

    return ranked_signal


def intra_industry_mean(x: pd.Series, industry: pd.Series):
    """
    Calculates intra-industry mean values for a signal.
    Parameters:
        x (pd.Series): Series of the signal to calculate intra-industry means for.
        industry (pd.Series): Series indicating the industry of each observation.
    Returns:
        pd.Series: Intra-industry mean values of the signal.
    """
    # Calculate intra-industry mean
    mean = x.groupby(industry).transform("mean")

    return mean


def intra_industry_std(x: pd.Series, industry: pd.Series):
    """
    Calculates intra-industry standard deviations for a signal.
    Parameters:
        x (pd.Series): Series of the signal to calculate intra-industry standard deviations for.
        industry (pd.Series): Series indicating the industry of each observation.
    Returns:
        pd.Series: Intra-industry standard deviations of the signal.
    """
    # Calculate intra-industry standard deviation
    sigma = x.groupby(industry).transform("std")

    return sigma


def intra_industry_quantiles(x: pd.Series, industry: pd.Series, q=0.1):
    """
    Isolates values in the bottom and top quantiles of a signal within each industry.
    Parameters:
        x (pd.Series): Series of the signal to isolate quantiles for.
        industry (pd.Series): Series indicating the industry of each observation.
        q (float, optional): Quantile threshold (default: 0.1).
    Returns:
        pd.Series: Intra-industry signal with values outside the quantiles set to NaN.
    """
    # Group data by industry
    industry_grouped_signal = x.groupby(industry)

    # Calculate quartile values for each industry
    lower_quantiles = industry_grouped_signal.transform(lambda x: np.quantile(x, q))
    upper_quantiles = industry_grouped_signal.transform(lambda x: np.quantile(x, (1 - q)))

    # Isolate values in the bottom and top quantiles within each industry
    quantile_signal = x.where((x <= lower_quantiles) | (x >= upper_quantiles)).dropna()

    return quantile_signal


def intra_industry_z_score(x: pd.Series, industry: pd.Series):
    """
    Calculates industry-neutral Z-scores for a raw signal and ensures dollar-neutrality.
    Parameters:
        x (pd.Series): Series of the signal to calculate Z-scores for.
        industry (pd.Series): Series indicating the industry of each stock.
    Returns:
        pd.Series: Industry-neutral Z-scores of the signal.
    Raises:
        AssertionError: If the sum of Z-scores within each industry is not close to zero (dollar-neutrality check).
    """

    # Calculate intra-industry ranked signal mean and standard deviation
    industry_mean = intra_industry_mean(x=x, industry=industry)
    industry_sigma = intra_industry_std(x=x, industry=industry)

    # Standardize intra-industry ranked signals
    z_score = (x - industry_mean) / industry_sigma

    # Check for dollar-neutrality
    assert -1e-10 < z_score.groupby(industry).sum().sum() < 1e-10

    # Industry dollar-neutral alpha model views
    return z_score


def intra_industry_ranked_signal_z_score(x: pd.Series, industry: pd.Series):
    """
    Calculates industry-neutral Z-scores for a ranked signal and ensures dollar-neutrality.
    Parameters:
        x (pd.Series): Series of the signal to calculate Z-scores for.
        industry (pd.Series): Series indicating the industry of each stock.
    Returns:
        pd.Series: Industry-neutral Z-scores of the ranked signal.
    Raises:
        AssertionError: If the sum of Z-scores within each industry is not close to zero (dollar-neutrality check).
    """

    # Create intra-industry ranked signals
    ranked_signal = intra_industry_rank(x=x, industry=industry)

    # Get the intra-industry z-score of ranked signals
    z_score = intra_industry_z_score(x=ranked_signal, industry=industry)

    # Check for dollar-neutrality
    assert -1e-10 < z_score.groupby(industry).sum().sum() < 1e-10

    # Industry dollar-neutral alpha model views
    return z_score


def beta_neutralize(views: pd.Series, betas: pd.Series) -> pd.Series:
    """
    Orthogonalizes alpha model to the market factor. Transforms alpha
    model views to beta-neutralized weights based on stock-level betas.
    These alpha weights can then be scaled to target desired level of vol.
    This function will eventually be extended to orthogonalize alpha model
    to other market factors (e.g., Fama-French factors).
    Parameters:
        views (pd.Series): Series of your alpha model views.
        betas (pd.Series): Series of ex-ante stock-level betas.
    Returns:
        pd.Series: Alpha weights for beta-neutralized views.
    """

    # Preprocess data
    X = sm.add_constant(betas).astype(float)
    y = views.astype(float)

    # Regress model views on stock-level betas
    model = sm.OLS(endog=y, exog=X).fit()
    alpha = model.params[0]
    beta = model.params[1]
    resid = y - (beta * X["ExAnte_Beta"] + alpha)

    # Orthogonalize views to create your alpha weights
    alpha_views = resid
    w = alpha_views / np.sum(np.abs(alpha_views))

    # Ensure dollar neturality
    assert -1e-10 < np.sum(w) < 1e-10

    return w


def industry_beta_neutralize(views: pd.Series, industry: pd.Series, betas: pd.Series):
    """
    Creates industry beta-neutral alpha weights for views based on stock-level betas and industries.
    Views are expected to be industry dollar-neutral before passed to this function... if views are not
    dollar neutral beforehand, there is no guaruntee that weights will be industry-dollar neutral.
    Parameters:
        views (pd.Series): Series of your alpha model views.
        industry (pd.Series): Series indicating the industry of each stock.
        betas (pd.Series): Series of ex-ante stock-level betas.
    Returns:
        pd.Series: Alpha weights for industry beta-neutralized views.
    """

    # Create industry dummy variables
    industry_dummies = pd.get_dummies(industry, prefix="Industry")

    print(industry_dummies.shape, betas.shape)

    # Create interaction terms: ExAnte_Beta * Industry_Dummies
    interaction_terms = pd.DataFrame(index=industry_dummies.index, columns=industry_dummies.columns)
    for i, beta in betas.items():
        interaction_terms.loc[i] = industry_dummies.loc[i] * beta

    # Beta Neutralize beta within industry
    w = beta_neutralize(views=views, betas=interaction_terms)

    return w


def mean_a(data: pd.DataFrame, a: int, column: str = "Adj Close") -> pd.DataFrame:
    """Rolling Mean; with period 'a'"""
    try:
        return data["Adj Close"].rolling(window=a).mean().dropna()
    except:
        raise Exception(f"Column {column} is not in {data}")


def plus(data_a: pd.DataFrame, data_b: pd.DataFrame) -> pd.DataFrame:
    """Sum of two dataframes"""
    try:
        return data_a + data_b
    except:
        raise Exception("The summands are not the same datatypes and cannot be summed")


def minus(data_a: pd.DataFrame, data_b: pd.DataFrame) -> pd.DataFrame:
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
