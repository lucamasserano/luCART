import pytest
import random

from luCART.classification_tree import impurity


def test_bayes_error():

    assert impurity.bayes_error(1) == 0
    assert impurity.bayes_error(0) == 0
    # assert impurity.bayes_error(random.uniform(0, 1)) == 0


def test_cross_entropy():

    eps = 1e-13
    assert impurity.cross_entropy(1) <= eps
    assert impurity.cross_entropy(0) <= eps
    # assert impurity.cross_entropy(random.uniform(0, 1)) == 0


def test_gini_index():

    assert impurity.gini_index(1) == 0
    assert impurity.gini_index(0) == 0
    # assert impurity.gini_index(random.uniform(0, 1)) == 0
