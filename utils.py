import lasagne.layers
import numpy as np
import theano.tensor as T

_EPSILON = np.finfo(np.float32).eps


def set_epsilon(eps):
    """
    test test
    """
    global _EPSILON
    _EPSILON = eps


def epsilon():
    """
    test
    """
    return _EPSILON
