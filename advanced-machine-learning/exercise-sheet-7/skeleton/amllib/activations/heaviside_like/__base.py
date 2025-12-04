"""Abstract base class implementation for Heaviside-like activation functions
   Protected in the heaviside sub package
"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.heviside_like.__base"
__package__ = "amllib.activations.heaviside_like"


from abc import ABCMeta, abstractmethod

import numpy as np

from ..activation import Activation


class HeavisideLike(Activation, metaclass=ABCMeta):
    """
    Abstract base class for all Heaviside-like
    activation functions.

    Attributes
    ----------
    factor : float
        Scaling factor for weight intialization.
        This factor is shared by all Heaviside like
        activation functions.
    """

    factor: float

    def __init__(self):
        """
            Initialize the base Heaviside function
            class with the shared weight initialization
            factor.
        """

        # Weight initialization factor.
        self.factor = 1.0
