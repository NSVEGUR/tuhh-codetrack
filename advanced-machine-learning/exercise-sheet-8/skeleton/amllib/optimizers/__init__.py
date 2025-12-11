"""Optimizer for loss minimization"""

__author__  = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.optimizers"
__package__ = "amllib.optimizers"

from .sgd import SGD
from .adam import Adam
from .optimizer import Optimizer
