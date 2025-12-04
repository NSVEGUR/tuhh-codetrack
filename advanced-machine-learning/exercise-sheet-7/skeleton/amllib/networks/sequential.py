"""Implementation of a sequential network"""

__author__ = 'Jens-Peter M. Zemke, Jonas Grams'
__version__ = '1.1'

__name__ = 'amllib.networks.sequential'
__package__ = 'amllib.networks'

import numpy as np

from amllib.layers import Layer
from amllib.losses import Loss, CrossEntropy


class Sequential:
    """
    Class representing a sequential network.
    """

    def __init__(self, input_shape: tuple[int, ...],
                 layers: list[Layer] = None, loss: Loss = None):
        """
        Initialize the sequential network.

        The sequential network is initialized with a given input shape.
        By default the list of layers is empty. If it is not, the output
        shape is determined by the last layer.

        Parameters
        ----------
        input_shape : tuple[int,...]
            Input shape (without the batch dimension).
        layers : list[Layer], optional
            List of layers of the network, by default empty.
        loss : Loss
            Loss for the training of the network. Default value: a new
            CrossEntropyLoss instance assuming SoftMax activation in the
            last layer.
        """

        if loss is None:
            self.loss = CrossEntropy()
        else:
            self.loss = loss

        if layers is None:
            self.layers = []
            self.input_shape = input_shape
            self.output_shape = input_shape
        else:
            self.layers = layers
            self.input_shape = input_shape
            self.output_shape = layers[-1].get_output_shape()

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Evaluate the sequential network layer-wise.

        Each layer in the layers list is evaluated with
        the output of the last layer as input.

        Parameters
        ----------
        input : np.ndarray
            Input for the first layer.

        Returns
        -------
        np.ndarray
            Result of the evaluation of the last layer in
            the layers list.
        """

        x = input
        for layer in self.layers:
            x = layer(x)

        return x

    def feedforward(self, input: np.ndarray) -> np.ndarray:
        """
        Evaluate the sequential network layer-wise while training.

        Each layer in the layers list is evaluated with
        the output of the last layer as input.

        Parameters
        ----------
        input : np.ndarray
            Input for the first layer.

        Returns
        -------
        np.ndarray
            Result of the evaluation of the last layer in
            the layers list.
        """

        x = input
        for layer in self.layers:
            x = layer.feedforward(x)

        return x

    def backprop(self, input: np.ndarray, labels: np.ndarray) -> float:
        """
        Backpropate backwards through the list of layers.

        Go backwards through the list of layers and compute
        the updates for all parameters (weights/filter banks and biases).
        For this the backprop method of each layer is used.

        Returns the mean loss of the batch.

        Parameters
        ----------
        input : np.ndarray
            Input data for the backpropagation.
        labels : np.ndarray
            Labels for the input data.

        Raises
        ------
        ValueError
            Raised if the input data shape does not match the input shape
            of the network.
        ValueError
            Raised if the labels shape does not match the output shape of
            the network.

        Returns:
            float: Loss of the batch computed with the networks loss function.
        """

        if input.shape[1:] != self.input_shape:
            raise ValueError("The input shape does not match the output shape"
                             f"of the network! Shape {self.input_shape} was "
                             f"expected, but it was {input.shape[1:]}")

        if labels.shape[1:] != self.output_shape:
            raise ValueError("The labels shape does not match the output shape"
                             f"of the network! Shape {self.output_shape} was"
                             f"expected, but it was {labels.shape[1:]}")

        # Compute loss and derivative of the loss function by the output of
        # the network as intial input for the backpropagation method of the
        # last layer.
        y = self.feedforward(input)
        loss = self.loss(y, labels)

        n_batch, _ = input.shape
        delta = self.loss.backprop(y, labels) / n_batch

        for layer in reversed(self.layers):

            delta = layer.backprop(delta)

        return loss

    def train(self, train_input: np.ndarray, train_labels: np.ndarray,
              batch_size: int = 1, epochs: int = 10,
              verbose: bool = True) -> np.ndarray:
        """
        Train the sequential network.

        The network is trained with the given training data. In each epoch
        backpropagation is used to update the parameters of each layer.

        Parameters
        ----------
        train_input : np.ndarray
            Input data for the training.
        train_labels : np.ndarray
            Labels for each input.
        batch_size : int, optional
            Batch size for training, by default 1
        epochs : int, optional
            Number of epochs to train, by default 10
        verbose : bool, optional
            Set to True if the training progress should be logged.

        Returns
        -------
        None|np.ndarray
            Average loss in each epoch.
        """

        n_data = train_labels.shape[0]
        n_batches = int(np.ceil(n_data/batch_size))

        epoch_losses = np.zeros(epochs)
        batch_losses = np.zeros(n_batches)

        for e in range(epochs):

            p = np.random.permutation(n_data)

            for j in range(n_batches):
                if verbose:
                    print(f'Epoch {e+1:>{len(str(epochs))}d}/{epochs}, '
                          f'Batch {j+1:>{len(str(n_batches))}d}/{n_batches}',
                          end='\r')

                batch_losses[j] = self.backprop(
                        train_input[p[j*batch_size:(j+1)*batch_size], :],
                        train_labels[p[j*batch_size:(j+1)*batch_size], :]
                    )

                for layer in self.layers:
                    layer.update()

            epoch_losses[e] = np.mean(batch_losses)

        if verbose:
            print('')

        return epoch_losses

    def add_layer(self, layer: Layer):
        """
        Add a layer to the list of layers.

        A layer is added to the list of layers
        and the output shape of the network is
        updated.

        Parameters
        ----------
        layer : Layer
            Layer to add.
        """
        # Build the layer with the old output shape of the network,
        # if not done yet.
        if not layer.built:
            layer.build(self.output_shape)

        self.layers.append(layer)
        self.output_shape = layer.get_output_shape()
