"""Test of the implementation of a sequential network w/ regularizer"""

from typing import Optional

import numpy as np

from amllib.networks import Sequential
from amllib.layers import Dense, Dropout
from amllib.losses import CrossEntropy
from amllib.activations import ReLU, Linear, Logistic, SoftMax, TanH
from amllib.utils import mnist
from amllib.optimizers import SGD, Adam, Optimizer
from amllib.regularizers import L2Regularizer


def get_model(
    hidden_optimizer: Optimizer,
    output_optimizer: Optimizer,
    regularization: bool = False) -> Sequential:
    """
    Generate a sequential FNN with one hidden layer of 100
    neurons.

    A FNN for the MNIST dataset with one hidden layer is generated.
    The number of neurons in the hidden layer is set to 100,
    the activation function is set to ReLU. The activation function
    of the output layer is the SoftMax activation function. The optimizer
    for the weight and bias updates depends on the argument `optimizer`.

    Parameters
    ----------
    hidden_optimizer : Optimizer
        Optimizer for the hidden layer
    output_optimizer : Optimizer
        Optimizer for the output layer
    regularization : bool
        Flag for L2 regularization.
        If True, L2 regularization is added to the layers.

    Returns
    -------
    Sequential
        A Sequential FNN for the MNIST dataset.
    """

    regularizer1 = L2Regularizer(0.005) if regularization else None
    regularizer2 = L2Regularizer(0.005) if regularization else None

    model = Sequential(input_shape=(784,), loss=CrossEntropy())
    model.add_layer(
        Dense(100,
              afun=ReLU(),
              optim=hidden_optimizer,
              kernel_regularizer=regularizer1))
    model.add_layer(
        Dense(10,
              afun=SoftMax(),
              optim=output_optimizer,
              kernel_regularizer=regularizer2))

    return model

def evaluate_model(model, x_train, y_train, x_test, y_test):

    y_tilde = model(x_train)
    prod = mnist.encode_labels(y_train) * np.log(y_tilde)
    loss = -np.sum(np.sum(prod, axis=1)) / y_tilde.shape[0]

    y_tilde = np.argmax(y_tilde, axis=1)
    accuracy = np.sum(y_tilde == y_train) / y_tilde.size

    print(f'Training loss: {loss:3.2e}')
    print(f'Training accuracy: {(accuracy * 100):5.2f}%')

    y_tilde = model(x_test)
    prod = mnist.encode_labels(y_test) * np.log(y_tilde)
    loss = -np.sum(np.sum(prod, axis=1)) / y_tilde.shape[0]

    y_tilde = np.argmax(y_tilde, axis=1)
    accuracy = np.sum(y_tilde == y_test) / y_tilde.size
    print(f'Test loss: {loss:3.2e}')
    print(f'Test accuracy: {(accuracy * 100):5.2f}%')

def test_regularizer():

    # define hyperparameters batch size and epochs
    bs, ep = 100, 20

    # Load MNIST dataset
    x_train, y_train, x_test, y_test = mnist.load_data()
    x_train = mnist.flatten_input(x_train)
    x_test = mnist.flatten_input(x_test)
    y = mnist.encode_labels(y_train)

    # define lists of optimizers for the test
    hidden_optimizers = [SGD(.2), SGD(.3, momentum=.9), Adam()]
    output_optimizers = [SGD(.2), SGD(.3, momentum=.9), Adam()]

    for opt1, opt2 in zip(hidden_optimizers, output_optimizers):

        print('--------------------------------------------')
        print('TEST', opt1.name, 'WITHOUT L2 REGULARIZATION')

        model = get_model(hidden_optimizer=opt1,
                          output_optimizer=opt2,
                          regularization=False)

        model.train(x_train, y, batch_size=bs, epochs=ep)
        evaluate_model(model, x_train, y_train, x_test, y_test)

        print('--------------------------------------------')
        print('TEST', opt1.name, 'WITH L2 REGULARIZATION')
        model = get_model(hidden_optimizer=opt1,
                          output_optimizer=opt2,
                          regularization=True)

        model.train(x_train, y, batch_size=bs, epochs=ep)
        evaluate_model(model, x_train, y_train, x_test, y_test)

if __name__ == '__main__':

    test_regularizer()
