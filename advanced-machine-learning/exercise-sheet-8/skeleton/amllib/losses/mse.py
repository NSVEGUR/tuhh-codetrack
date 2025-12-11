"""
Implementation of the mean squared error loss function.
"""

__author__ = 'Jens-Peter M. Zemke, Jonas Grams'
__version__ = '1.0'

__name__ = 'amllib.losses.mse'
__package__ = 'amllib.losses'

import numpy as np
from numpy.typing import ArrayLike

from amllib.losses import Loss


class MeanSquaredError(Loss):
    """
    Implementation of the meann squared error loss.
    This is computed as scaled euclidean error between
    the  outputs of a network and the labels.
    """

    def __call__(self, network_output: ArrayLike,
                 target_output: ArrayLike) -> float:
        """
        Calculate the mean squared error loss for given network outputs
        and target outputs. It is computed by

        $$
            C_{\\mathrm{MSE}}(\\widetilde{\\mathbf{y}}) = \\frac{1}{2}
            \\|\\widetilde{\\mathbf{y}} - \\mathbf{y}\\|^{2},
        $$

        where $\\widetilde{\\mathbf{y}}$ are the network outputs for a single
        input and $\\mathbf{y}$ is the target output for this input.

        Parameters
        ----------
        network_output : ArrayLike
            Output of a network which is compared to the target output.
        target_output : ArrayLike
            Target output.

        Returns
        -------
        float
            Mean squared error loss for the given (batch of) network outputs
            and target output
        """
        diff = np.subtract(network_output, target_output)
        norm_diff = np.linalg.norm(diff, axis=1, ord=2)
        losses = 0.5 * norm_diff**2
        return np.mean(losses, axis=0)

    def backprop(self, network_output: ArrayLike,
                 target_output: ArrayLike) -> np.ndarray:
        """
        Compute the derivatives of the mean squared error loss for given
        network outputs and target outputs. It is given by

        $$
        \\frac{\\partial C_{\\mathrm{MSE}}}{\\partial \\widetilde{\\mathbf{y}}}
        (\\widetilde{\\mathbf{y}}, \\mathbf{y})
        = \\widetilde{\\mathbf{y}} - \\mathbf{y},
        $$

        where $\\widetilde{\\mathbf{y}}$ are the network outputs for a single
        input and $\\mathbf{y}$ is the target output for this input.

        Parameters
        ----------
        network_output : ArrayLike
            Output of a network which is compared to the target output.
        target_output : ArrayLike
            Target outputs.

        Returns
        -------
        float
            Derivatives of the mean squared error loss for the given
            (batch of) network outputs and target outputs.
        """

        diff = np.subtract(network_output, target_output)
        return diff
