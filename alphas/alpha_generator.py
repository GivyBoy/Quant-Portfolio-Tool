"""
Controls the generation of the alpha returns
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import alpha_functions  # module that houses all the alpha functions
from alpha_builder import FormulaicAlphaBuilder  # builds the alphas

# Local imports

sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent.absolute()))

import pandas as pd

from data import data_utils

DB_DATA = data_utils.load_cache(r"data/sp_500/sp_500_cache.pickle")
RETURNS_DATA = pd.DataFrame(data_utils.load_cache(r"data/sp_500/adj_close_returns.pickle"))

# `dir(alpha_functions)` returns all functions within the specified file,
# afterwich all the alpha functions are filtered out, for the alphas that
# start with `alpha`. All custom alphas will have the `custom` prefix
alpha_funcs = [func for func in dir(alpha_functions) if func.startswith("alpha")]
alpha_list = list(
    map(
        lambda alpha_func: FormulaicAlphaBuilder(
            alpha_func=getattr(alpha_functions, alpha_func), start_date="19991231", vol_target=0.2
        ),
        alpha_funcs,
    )
)

alpha_list.append(
    # this is used to manually instantiate the classes for signals that
    # have customized components
    FormulaicAlphaBuilder(
        alpha_func=alpha_functions.alpha_01,
        start_date="19991231",
        vol_target=0.25,
        get_data=(DB_DATA, RETURNS_DATA),
    ),
)
