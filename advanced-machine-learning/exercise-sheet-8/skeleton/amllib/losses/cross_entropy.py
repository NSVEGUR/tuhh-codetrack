"""
Implementation of the cross entropy loss
"""

__author__ = 'Jens-Peter M. Zemke, Jonas Grams'
__version__ = '1.0'

__name__ = 'amllib.lossese.cross_entropy'
__package__ = 'amllib.loss'

import numpy as np
from numpy.typing import ArrayLike

from amllib.losses import Loss


class CrossEntropy(Loss):

    def __call__(self, network_output: ArrayLike,
                 target_output: ArrayLike) -> float:
        """
        Calculate the cross entropy loss for given network outputs
        and target outputs. It is computed by

        $$
            C_{\\mathrm{CE}}(\\widetilde{\\mathbf{y}, \\mathbf{y}} =
            -\\sum_{i=0}^{n} y_i \\log(\\widetilde{y}_i),
        $$

        where $\\widetilde{\\mathbf{y}}\\in\\mathbb{R}^n$ are the network
        outputs for a single input and $\\mathbf{y}\\in\\mathbb{R}^{n}$ is
        the target output for this input.

        Parameters
        ----------
        network_output : ArrayLike
            Output of a network which is compared to the target output.
        target_output : ArrayLike
            Target output.

        Returns
        -------
        float
            Cross entropy loss for the given (batch of) network outputs
            and target output
        """
        product = np.multiply(target_output, np.log(network_output))
        losses = -np.sum(product, axis=1)
        return np.mean(losses, axis=0)

    def backprop(self, network_output, target_output):
        """
        Compute the derivatives of the cross entropy loss for given
        network outputs and target outputs assuming the activation of
        the output layer is the SoftMax activation function.
        It is given by

        $$
        \\frac{\\partial C_{\\mathrm{CE}}}{\\partial \\widetilde{\\mathbf{y}}}
        (\\widetilde{\\mathbf{y}}, \\mathbf{y})
        = \\widetilde{\\mathbf{y}} - \\mathbf{y},
        $$

        where $\\widetilde{\\mathbf{y}}$ are the network outputs for a single
        input and $\\mathbf{y}$ is the target output for this input
        (See exercise 1 on exercise sheet 4).

        Parameters
        ----------
        network_output : ArrayLike
            Output of a network which is compared to the target output.
        target_output : ArrayLike
            Target outputs.

        Returns
        -------
        float
            Derivative of the mean squared error loss for the given
            (batch of) network outputs and target outputs.
        """
        return np.subtract(network_output, target_output)
