"""Weight matrix initialization with random orthogonal matrices."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.initializers.randn_orthonormal"
__package__ = "amllib.initializers"


import numpy as np
from numpy.random import randn
from numpy.linalg import svd

from .initializer import KernelInitializer

class RandnOrthonormal(KernelInitializer):
  """
  Class for random orthonormal weight initialization.

  This class initializes weight matrices as orthonormal matrices.
  """

  def wfun(self, m: int, n: int) -> np.ndarray:
    """
    Initialize a random orthonormal weight matrix.

    A random orthonormal weight matrix of shape
    $(m, n)$ is generated with the standard normal
    distribution.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    np.ndarray
        Random orthonormal weight matrix of shape `(m, n)`.
    """
    A = randn(m, n)
    U, _, V = svd(A, full_matrices=False)
    if n >= m:
      return V
    else:
      return U

  def ffun(self, m: int, c: int, fh: int, fw: int):

    return super().ffun(m, c, fh, fw)
