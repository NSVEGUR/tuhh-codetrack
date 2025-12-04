import importlib
import importlib.resources
import numpy as np

from amllib.activations import ReLU
from amllib.networks import FeedforwardNet
from amllib.utils import mnist

if __name__ == '__main__':

    # load and scale MNIST data
    x_train, y_train, x_test, y_test = mnist.load_data()

    # flatten 3-tensors to matrices
    x_train = mnist.flatten_input(x_train)
    x_test = mnist.flatten_input(x_test)

    # one-hot encode labels
    labels = mnist.encode_labels(y_train)

    # define neural network
    model = FeedforwardNet([784, 100, 10], ReLU)

    # train neural network
    model.train(x_train.T, labels.T,
                batch_size=10,
                epochs=10,
                learning_rate=.02)

    # compute accuracy on test data set
    y_tilde = np.argmax(model(x_test.T), axis=0)
    accuracy = np.sum(y_tilde == y_test) / 10000
    print(f'Accuracy: {accuracy * 100}%')
