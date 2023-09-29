"""
Utility methods to download and/or cache data
"""
from __future__ import annotations

import os

import data_utils

CWD = os.path.dirname(os.path.abspath(__file__))

# Define constants
R3K_TICKER_PATH = os.path.join(CWD, r"r3k_tickers.txt")
R3K_CACHE_PATH = os.path.join(CWD, r"russell_3000/russell_3000_cache.pickle")
SP_500_CACHE_PATH = os.path.join(CWD, r"sp_500/sp_500_cache.pickle")

UNIVERSE_LIST = ["russell_3000", "sp_500"]


def main():
    """
    Runs data-injestion, pre-processing, and caching process for a user-specified
    universe.
    """

    # Get user-specified universe
    universe = "russell_3000"

    assert universe in UNIVERSE_LIST, universe

    # Define cross universe cache path
    cross_universe_cache_path = os.path.join(CWD, rf"{universe}")

    if universe == "russell_3000":
        # Get list of stock tickers
        f = open(R3K_TICKER_PATH, "r")
        ticker_list = f.read().splitlines()

        # Get OHLC + returns data
        ohlc_data = data_utils.get_ohlc_data(ticker_list)

        # Ensure data was acquired for each ticker
        try:
            assert list(ohlc_data.keys()) == ticker_list
        except:
            for key in ohlc_data.keys():
                if key not in ticker_list:
                    print(key)

        # Cache data to database
        data_utils.cache(data=ohlc_data, path=R3K_CACHE_PATH)

        # Cache in cross-universe manner
        data_utils.cache_cross_universe_statistics(ohlc_data=ohlc_data, path=cross_universe_cache_path)

    elif universe == "sp_500":
        # Get list of stock tickers
        ticker_list = data_utils.get_sp500_tickers()

        # Get OHLC + returns data
        ohlc_data = data_utils.get_ohlc_data(ticker_list)

        # Ensure data was acquired for each ticker
        try:
            assert list(ohlc_data.keys()) == ticker_list
        except:
            for key in ohlc_data.keys():
                if key not in ticker_list:
                    print(key)

        # Cache data to database
        data_utils.cache(data=ohlc_data, path=SP_500_CACHE_PATH)

        # Cache in cross-universe manner
        data_utils.cache_cross_universe_statistics(ohlc_data=ohlc_data, path=cross_universe_cache_path)


if __name__ == "__main__":
    main()
