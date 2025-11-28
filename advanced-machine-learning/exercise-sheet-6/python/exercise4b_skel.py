"""
Exercise 4b of exercise sheet 6.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import History
from matplotlib.axes import Axes


def get_fnn_model(p1: float = 0.5, p2: float = 0.2):

    model = tf.keras.Sequential()

    model.add(
        tf.keras.Input(shape=(28*28,), name='fnn_input')
    )
    model.add(
        tf.keras.layers.Dense(256, activation='relu', name='fnn_dense_1')
    )
    model.add(tf.keras.layers.Dropout(p1, name='fnn_dropout_1'))
    model.add(
        tf.keras.layers.Dense(128, activation='relu', name='fnn_dense_2')
    )
    model.add(tf.keras.layers.Dropout(p2, name='fnn_dropout_2'))
    model.add(
        tf.keras.layers.Dense(10, name='fnn_output')
    )

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True), optimizer='adam', metrics=['accuracy'])

    return model


def get_cnn_model(p1: float = 0.5, p2: float = 0.2):

    # 28x28 RGB (3 channels) images
    input_shape = (28, 28, 1)

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=input_shape))

    # CNN part of the network
    model.add(tf.keras.layers.Conv2D(
        32, kernel_size=(3, 3), activation='relu', name='cnn_conv2d_1'))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), name='cnn_max_pooling2d_1'))
    model.add(tf.keras.layers.Conv2D(
        64, kernel_size=(3, 3), activation='relu', name='cnn_conv2d_2'))
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2), name='cnn_max_pooling2d_2'))
    model.add(tf.keras.layers.Dropout(p1, name='cnn_dropout_1'))

    # Dense classification part of the network
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        128, activation='relu', name='cnn_dense_1'))
    model.add(tf.keras.layers.Dropout(p2, name='cnn_dropout_2'))
    model.add(tf.keras.layers.Dense(10, name='cnn_output'))

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(
        from_logits=True), optimizer='adam', metrics=['accuracy'])

    return model


def train_fnn_fmnist_model(model, epochs) -> History:

    # Load fashion_mnist_data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = np.reshape(x_train, (-1, 28*28))
    x_train = x_train.astype('float32') / 255.0

    x_test = np.reshape(x_test, (-1, 28*28))
    x_test = x_test.astype('float32') / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Train the model with the fashion_mnist dataset
    history = model.fit(x_train, y_train, validation_data=(
        x_test, y_test), batch_size=128, epochs=epochs)

    return history


def train_cnn_fmnist_model(model, epochs: int = 30) -> History:

    # Load fashion_mnist_data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.0

    x_test = x_test.astype('float32') / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    history = model.fit(x_train, y_train, validation_data=(
        x_test, y_test), batch_size=128, epochs=epochs)

    return history


def plot_histories(ax_loss: Axes, ax_acc: Axes,
                   fnn_history: History, cnn_history: History):
    """
    Plot the training and test losses and training and test accuracies
    stored in `fnn_history` and `cnn_history`.
    """
    epochs = len(cnn_history.history['loss'])

    cnn_color = '#1f77b4'  # Blue for cnns
    fnn_color = '#ff7f0e'  # Orange for fnns

    ax_loss.plot(np.arange(epochs), cnn_history.history['loss'],
                 linestyle='-', color=cnn_color, label='CNN - train loss')
    ax_loss.plot(np.arange(epochs), cnn_history.history['val_loss'],
                 linestyle='--', color=cnn_color, label='CNN - test loss')
    ax_loss.plot(np.arange(epochs), fnn_history.history['loss'],
                 linestyle='-', color=fnn_color, label='FNN - train loss')
    ax_loss.plot(np.arange(epochs), fnn_history.history['val_loss'],
                 linestyle='--', color=fnn_color, label='FNN - test loss')

    ax_loss.legend()  # Generate legend
    ax_loss.set_yscale('log')  # Set logarithmic scale for y-axis
    ax_loss.set_xlabel('Epochs', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.grid(which='both', linestyle='--', linewidth=.5)

    ax_acc.plot(np.arange(epochs), cnn_history.history['accuracy'],
                linestyle='-', color=cnn_color, label='CNN - train accuracy')
    ax_acc.plot(np.arange(epochs), cnn_history.history['val_accuracy'],
                linestyle='--', color=cnn_color, label='CNN - test accuracy')
    ax_acc.plot(np.arange(epochs), fnn_history.history['accuracy'],
                linestyle='-', color=fnn_color, label='FNN - train accuracy')
    ax_acc.plot(np.arange(epochs), fnn_history.history['val_accuracy'],
                linestyle='--', color=fnn_color, label='FNN - test accuracy')

    ax_acc.legend()  # Generate legend
    ax_acc.set_xlabel('Epochs', fontsize=12)
    ax_acc.set_ylabel('Accuracy', fontsize=12)
    ax_acc.grid(which='both', linestyle='--', linewidth=.5)


if __name__ == '__main__':

    # TODO set the dropout probabilities here
    fnn_p1, fnn_p2 = 0.0, 0.0
    cnn_p1, cnn_p2 = 0.0, 0.0

    # If the training takes too long, reduce the number of epochs here.
    epochs = 30

    print('-----------------------------------------')
    print('               FNN Model                 ')
    print('-----------------------------------------')
    fnn_model = get_fnn_model(fnn_p1, fnn_p2)
    fnn_model.summary()

    print('-----------------------------------------')
    print('          Training FNN Model             ')
    print('-----------------------------------------')
    fnn_history = train_fnn_fmnist_model(fnn_model, epochs)

    print('-----------------------------------------')
    print('               CNN Model                 ')
    print('-----------------------------------------')
    cnn_model = get_cnn_model(cnn_p1, cnn_p2)
    cnn_model.summary()

    print('-----------------------------------------')
    print('          Training CNN Model             ')
    print('-----------------------------------------')
    cnn_history = train_cnn_fmnist_model(cnn_model, epochs)

    # Create figure with 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    plot_histories(ax[0], ax[1], fnn_history, cnn_history)

    fig.savefig(f'fmnist_cnn_vs_fnn_{(cnn_p1, cnn_p2)}_{
                (fnn_p1, fnn_p2)}.pdf', bbox_inches='tight')
