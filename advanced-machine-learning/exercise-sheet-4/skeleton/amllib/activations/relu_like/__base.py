"""Abstract base class implementation for Heaviside-like activation functions
   Protected in the heaviside sub package
"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.relu_like.__base"
__package__ = "amllib.activations.relu_like"


from abc import ABCMeta, abstractmethod

import numpy as np

from ..activation import Activation


class ReLULike(Activation, metaclass=ABCMeta):
    """
    Abstract base class for all ReLU like
    activation functions. All ReLU like
    activations share the same weight intialization
    factor.

    Attributes
    ----------
    factor: float
        Scaling factor for weight intialization.
        This factor is shared by all ReLU like
        activation functions.
    """

    def __init__(self):
        """
            Initialize the base ReLU function
            class with the shared weight initialization
            factor.
        """

        # Weight initialization factor.
        self.factor = np.sqrt(2.0)
