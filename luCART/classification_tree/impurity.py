import numpy as np
from typing import Union, Callable, Iterable


def impurity_reduction(impurity_function: Callable,
                       p: Union[float, np.ndarray],
                       p_left: Union[float, np.ndarray],
                       p_right: Union[float, np.ndarray],
                       n_obs_left: Union[float, np.ndarray], n_obs_right: Union[float, np.ndarray],
                       ) -> Union[float, np.ndarray]:

    return impurity_function(p) \
           - np.multiply(n_obs_left, impurity_function(p_left)) \
           - np.multiply(n_obs_right, impurity_function(p_right))


def bayes_error(p: Union[float, Iterable]) -> Union[float, np.ndarray]:
    return np.minimum(p, 1-p)


def cross_entropy(p):

    # cross_entropy is undefined for p = 1 or p = 0
    eps = 1e-15
    p = np.maximum(eps, np.minimum(1 - eps, p))

    return - np.multiply(p, np.log(p)) - np.multiply(1-p, np.log(1-p))


def gini_index(p):
    return np.multiply(p, 1-p)

# TODO: if necessary, define other measures of impurity
