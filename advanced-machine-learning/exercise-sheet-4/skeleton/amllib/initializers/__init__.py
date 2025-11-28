"""Package for weight matrix and filter bank initializers."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.initializers"
__package__ = "amllib.initializers"

from .rand_average import RandAverage
from .randn_average import RandnAverage
from .randn_orthonormal import RandnOrthonormal
from .initializer import KernelInitializer
