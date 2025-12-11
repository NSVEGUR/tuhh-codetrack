"""Abstract base class for Sign-like activation functions.
   Protected in the sign sub package
"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.heaviside_like.__base"
__package__ = "amllib.activations.heaviside_like"

from abc import ABCMeta, abstractmethod

import numpy as np

from ..activation import Activation


class SignLike(Activation, metaclass=ABCMeta):
    """
    Abstract base class for all Sign like
    activation functions. All Sign like
    activations share the same weight intialization
    factor.

    Attributes
    ----------
    factor: float
        Scaling factor for weight intialization.
        This factor is shared by all Sign like
        activation functions.
    """

    def __init__(self):
        """
            Initialize the base Sign function
            class with the shared weight initialization
            factor.
        """

        self.factor = 1.0
