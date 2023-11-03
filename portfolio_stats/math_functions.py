"""
This file holds all the math related statistical functions that will be used
in the project (this is a test file)
"""
from __future__ import annotations

import numpy as np


class MathFunctions:
    """
    This class calculates some simple statistics, when given an array of
    numbers
    """

    def __init__(self, numbers: list[int or float]) -> None:
        self.numbers = numbers

    def sum(self) -> int or float:
        """
        Returns the sum of the list of numbers
        """
        return np.sum(self.numbers)

    def mean(self) -> int or float:
        """
        Returns the mean of the list of numbers
        """
        if len(self.numbers) > 0:
            return np.mean(self.numbers)
        return 0

    def std(self) -> int or float:
        """
        Returns the standard deviation of the list of numbers
        """
        if len(self.numbers) > 0:
            return np.std(self.numbers)
        return 0
