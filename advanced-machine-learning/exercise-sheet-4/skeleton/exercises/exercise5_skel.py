import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from amllib.networks import Sequential
from amllib.initializers import RandnAverage
from amllib.layers import Dense
from amllib.losses import CrossEntropy
from amllib.activations import ReLU, SoftMax
from amllib.utils import mnist, fmnist


def get_mnist_model():

    model = Sequential(input_shape=(784,), loss=CrossEntropy())

    # TODO Add your MNIST model, i.e., the hidden layer(s), here.
    # For example, you can use the network from exercise sheet 2.

    model.add_layer(
        Dense(10, afun=SoftMax(), initializer=RandnAverage(),
              learning_rate=0.01)
    )

    return model


def get_fmnist_model():

    model = Sequential(input_shape=(784,), loss=CrossEntropy())

    # TODO Add your fashion MNIST model, i.e., the hidden layer(s), here

    model.add_layer(
        Dense(10, afun=SoftMax(), initializer=RandnAverage(),
              learning_rate=0.001)
    )

    return model


def test_mnist_model():

    # Load MNIST dataset
    x_train, y_train, x_test, y_test = mnist.load_data()

    y = mnist.encode_labels(y_train)

    x_train = mnist.flatten_input(x_train)
    x_test = mnist.flatten_input(x_test)

    model = get_mnist_model()

    losses = model.train(x_train, y, batch_size=100, epochs=20)

    y_tilde = model(x_test)
    y_tilde = np.argmax(y_tilde, axis=1)

    accuracy = np.sum(y_tilde == y_test) / 10000

    print(f'Test accuracy: {(accuracy * 100):5.2f}%')

    return losses


def test_fmnist_model():

    # Load MNIST dataset
    x_train, y_train, x_test, y_test = fmnist.load_data()

    y = mnist.encode_labels(y_train)

    x_train = mnist.flatten_input(x_train)
    x_test = mnist.flatten_input(x_test)

    model = get_fmnist_model()

    losses = model.train(x_train, y, batch_size=100, epochs=20)

    y_tilde = model(x_test)
    y_tilde = np.argmax(y_tilde, axis=1)

    accuracy = np.sum(y_tilde == y_test) / 10000

    print(f'Test accuracy: {(accuracy * 100):5.2f}%')

    return losses


if __name__ == '__main__':

    print('----------------------------------')
    print('       Training MNIST model       ')
    print('----------------------------------')

    losses_mnist = test_mnist_model()

    print('----------------------------------')
    print('      Training FMNIST model       ')
    print('----------------------------------')

    losses_fmnist = test_fmnist_model()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot loss for the MNIST model
    ax.plot(losses_mnist, marker='o', lw=4, markersize=10,
            label='MNIST training loss')
    # Plot loss for the fashion MNIST model
    ax.plot(losses_fmnist, marker='o', lw=4, markersize=10,
            label='Fashion MNIST training loss')

    ################################
    #      Set plot parameters     #
    ################################

    # Create legend with fontsize 20pt for all texts
    ax.legend(fontsize=20)
    # Set major (labled) x-axis ticks at 0, 5, 10, 15, 20
    ax.set_xticks(range(0, 21, 5))
    # Set minor x-axis ticks for all integers between 0 and 20
    ax.set_xticks(range(21), minor=True)
    # Set fontsize for axis tick labels to 20pt
    ax.tick_params(axis='both', labelsize=20)
    # Set x axis label with fontsize 20pt
    ax.set_xlabel('Epochs', fontsize=20)
    # Set y axis label with fontsize 20pt
    ax.set_ylabel('Loss', fontsize=20)
    # Set logarithmic scale for y-axis
    ax.set_yscale('log')
    # Create grid with major ticks
    ax.grid(which='major')
    # Save plot as pdf file
    plt.savefig('MNIST_vs_fashion_MNIST.pdf', bbox_inches='tight')

    # Save losses for possible later processing
    np.savez('losses_mnist_fmnist.npz', losses_mnist=losses_mnist,
             losses_fmnist=losses_fmnist)
