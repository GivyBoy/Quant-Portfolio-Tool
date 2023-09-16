"""
Transforms for the raw signals
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from alpha_builder_utils import quantiles


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, data: pd.Series) -> list[pd.Series]:
        transformed = []
        for tranform in self.transforms:
            transformed.append(tranform(data))
        return transformed


class Zscore:
    def __call__(self, data: pd.Series) -> pd.Series:
        return (data - np.mean(data)) / np.std(data)


class Quantile:
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, data: pd.Series) -> pd.Series:
        return quantiles(data, self.p)


if __name__ == "__main__":
    zscore = Zscore()
    print(zscore(np.array([1, 2, 3, 4, 5])))
