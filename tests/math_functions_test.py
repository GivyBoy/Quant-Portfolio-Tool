"""
Tests for all functions in the math_functions.py file (this is a test file)
"""
from __future__ import annotations

from portfolio_stats.math_functions import MathFunctions


def test_math_functions_returns_correct_value():
    """
    Ensures that the MathFunctions class returns the correct values
    """
    numbers = [1, 2, 3, 4, 5, 6]
    zero = []

    test_func = MathFunctions(numbers=numbers)
    zero_func = MathFunctions(numbers=zero)

    assert test_func.sum() == 21
    assert test_func.mean() == 3.5
    assert round(test_func.std(), 2) == 1.71
    assert zero_func.sum() == 0
