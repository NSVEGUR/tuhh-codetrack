"""Abstract base class for Abs-like activation functions."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.abs_like.__base"
__package__ = "amllib.activations.abs_like"

from abc import ABCMeta, abstractmethod

import numpy as np

from ..activation import Activation


class AbsLike(Activation, metaclass=ABCMeta):
    """
    Abstract base class for all Abs-like
    activation functions. All Abs-like
    activations share the same weight intialization
    factor.

    Attributes
    ----------
    factor: float
        Scaling factor for weight intialization.
        This factor is shared by all Abs like
        activation functions.
    """

    def __init__(self):
        """
            Initialize the base Abs function
            class with the shared weight initialization
            factor.
        """

        self.factor = 1.0
