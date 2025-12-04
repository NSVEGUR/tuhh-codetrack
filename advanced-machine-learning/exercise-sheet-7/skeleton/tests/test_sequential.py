"""Test of the implementation of a sequential network"""

import numpy as np

from amllib.networks import Sequential
from amllib.layers import Dense
from amllib.losses import CrossEntropy, MeanSquaredError
from amllib.activations import ReLU, Linear, SoftMax
from amllib.optimizers import SGD
from amllib.utils import mnist

def get_model():

    # you might want to exchange SoftMax for Linear
    # and see what happens to the accuracy

    model = Sequential(input_shape=(784,), loss=CrossEntropy())
    model.add_layer(Dense(100,
                          afun=ReLU(),
                          optim=SGD(.2)))
    model.add_layer(Dense(10,
                          afun=SoftMax(),
                          optim=SGD(.2)))

    return model

def main():

    # Load MNIST dataset
    x_train, y_train, x_test, y_test = mnist.load_data()

    y = mnist.encode_labels(y_train)

    x_train = mnist.flatten_input(x_train)
    x_test = mnist.flatten_input(x_test)

    model = get_model()

    model.train(x_train, y, batch_size=100, epochs=10)

    y_tilde = model(x_test)
    y_tilde = np.argmax(y_tilde, axis=1)

    accuracy = np.sum(y_tilde == y_test) / 10000

    print(f'Test accuracy: {(accuracy * 100):5.2f}%')

if __name__ == '__main__':

    main()
