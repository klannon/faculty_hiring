# -*- coding: utf-8 -*-

import pytest
from faculty_hiring.skeleton import fib

__author__ = "Kevin Lannon"
__copyright__ = "Kevin Lannon"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
