"""Abstract base class for layers."""

__author__ = 'Jens-Peter M. Zemke, Jonas Grams'
__version__ = '1.1'

__name__ = 'amllib.layers.layer'
__package__ = 'amllib.layers'

from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    """
    Abstract base class for layers.

    This abstract base class defines common methods
    all neural network layers have to implement.

    Attributes
    ----------
    built: bool
        Flag to determine if the layer has been built.
        It should be set to `True` when the `build` method
        is called.
    """

    built = False

    @abstractmethod
    def build(self, input_shape: tuple[int, ...]) -> None:
        """
        Build the layer.

        Every layer should implement a build function. It sets the input shape
        and initializes all parameters.

        Parameters
        ----------
        input_shape : tuple[int,...]
            Shape of the input.
        """

        pass

    @abstractmethod
    def __call__(self, a: np.ndarray) -> np.ndarray:
        """
        Evaluate the layer.

        Parameters
        ----------
        a : np.ndarray
            Input array. It has to match the input shape (Without
            the batch dimension). If the layer has not been built,
            it should call the build method with the shape of the
            current input.

        Returns
        -------
        np.ndarray
            Evaluated input.
        """

        pass

    @abstractmethod
    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """
        Evaluate the layer.

        Evaluate the layer and cache data for backpropagation. This method
        should be used while training networks.

        Parameters
        ----------
        a : np.ndarray
            Input array. It has to match the input shape (Without
            the batch dimension). If the layer has not been built,
            it should call the build method with the shape of the
            current input.

        Returns
        -------
        np.ndarray
            Evaluated input.
        """

        pass

    @abstractmethod
    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Backpropagate through this layer.

        Every layer should implement a backpropagation method. It
        is called for training and should compute updates for all
        parameters.

        Parameters
        ----------
        delta : np.ndarray
            Input array. It contains the derivative of the used
            cost function by the output of the layer. With this
            the updates for all parameters can be computed.

        Returns
        -------
        np.ndarray
            Derivative of the cost function by the input. This is
            needed for the next layer.
        """

        pass

    @abstractmethod
    def get_output_shape(self) -> tuple[int, ...]:
        """
        Get the shape of the outputs of the layer.

        This method should return the shape of the
        outputs of the layer (without the batch dimension).

        Returns
        -------
        tuple[int,...]
            Output shape (without the batch dimension).
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """
        Update all parameters.

        This layer is responsible for updating all parameters.
        It is called in the training process.
        """
        pass
