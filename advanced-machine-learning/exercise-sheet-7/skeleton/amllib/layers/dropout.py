"""Implementation of a dropout layer."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.layers.dropout"
__package__ = "amllib.layers"

import numpy as np
from typing import Optional

from .layer import Layer


class Dropout(Layer):
    """
    Class representation of a dropout layer.

    This class represents dropout layers. It is supposed to be
    used after a dense or a convolutional layer. The evaluation
    of this layer simulates the random dropout of neuron of a layer.

    Attributes
    ----------
    p: float
        Dropout parameter between 0 and 1. A neuron is supposed
        to be kept with this probability. The dropping probability
        is therefore 1-`p`.
    train: bool
        Training flag used to distinguish between the evaluation
        for training and the evaluation when the network is not
        trained.
    mask: np.ndarray
        Information about dropped and kept neurons in the last evaluation.
        These information are used for backpropagation.
    input_shape: tuple[int, ...]
        Shape of the inputs (without the batch dimension).
    """

    def __init__(self, input_shape: Optional[tuple[int, ...]] = None, p: float = .5):
        """
        Initialize the dropout layer.

        Parameters
        ----------
        input_shape : Optional[tuple[int, ...]], optional
            Shape of the inputs, by default None. If the shape is not
            None, the layer is built on initialization.
        p : float, optional
            Dropout probability parameter, by default 0.5.
            A neuron is dropped with this probability.
            Has to be between 0.0 and 1.0.

        Raises
        ------
        ValueError
            Raised if the probability p is not between 0.0 and 1.0.
        """
        super().__init__()

        if not 0.0 <= p <= 1.0:
            raise ValueError("The probability parameter has to be between"
                             f"0.0 and 1.0, but it is {p}.")

        self.p = p
        self.mask = None

        if input_shape is not None:

            self.build(input_shape)

    def build(self, input_shape: tuple[int,...]) -> None:
        """
        Build the dropout layer.

        This method sets the input shape of the layer.

        Parameters
        ----------
        input_shape : tuple[int,...]
            Shape of the inputs.
        """
        self.input_shape = input_shape

        self.built = True

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Evaluate the Dropout layer.

        The Dropout layer is evaluated in non-training mode.
        The input is not altered.

        Parameters
        ----------
        input : np.ndarray
            Input array, has to match the input shape.

        Returns
        -------
        np.ndarray
            Output array, equals the input array.
        """
        return input

    def feedforward(self, input: np.ndarray) -> np.ndarray:
        """
        Evaluate the Dropout layer.

        Choose randomly dropped neurons and return an array
        filled with zeros where neurons were dropped.
        The drop pattern is cached for backpropagation.
        This method is intended for training.

        If the network is not trained, return a weighted
        output instead.

        Parameters
        ----------
        input : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Output array with either dropped neurons or a
            weighted input.
        """
        # Build the layer with the shape of the inputs, if not done yet.
        if not self.built:
            self.build(input.shape[1:])

        shape = (1,) + input.shape[1:]  # same mask for minibatch

        self.mask = 1 / (1-self.p) * \
          np.random.binomial(1, 1-self.p, size=shape)
        drop_a = input * self.mask
        return drop_a

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Backpropagate through the layer.

        Takes the previous computed derivative of the cost
        function by the output and computes the derivative
        of the cost function by the input.

        Parameters
        ----------
        delta : np.ndarray
            Previous computed derivative.

        Returns
        -------
        np.ndarray
            Derivative of the cost function by the input.
        """

        delta *= self.mask
        return delta

    def get_output_shape(self) -> tuple[int, ...]:
        """
        Get the shape of the output of the layer.

        Returns
        -------
        tuple[int, ...]
            Output shape.
        """

        return self.input_shape

    def update(self):
        """
        Update function of the Dropout layer.

        Since the layer has no trainable parameters,
        nothing has to be done here.
        """
        pass
