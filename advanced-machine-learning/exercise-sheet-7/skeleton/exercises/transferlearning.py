import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from os.path import dirname, join
from tensorflow.keras.utils import \
     image_dataset_from_directory as ds_from_dir
from tensorflow.keras.utils import get_file, plot_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

def get_cats_and_dogs(b_size, img_size):

    url = 'https://download.mlcc.google.com/' + \
      'mledu-datasets/cats_and_dogs_filtered.zip'
    path_to_zip = get_file(
        'cats_and_dogs.zip',
        origin=url,
        extract=True,
        cache_dir=".")

    path = join(path_to_zip, 'cats_and_dogs_filtered')
    train_dir = join(path, 'train')
    validation_dir = join(path, 'validation')

    train_ds = ds_from_dir(train_dir,
                           shuffle=True,
                           batch_size=b_size,
                           image_size=img_size)
    validation_ds = ds_from_dir(validation_dir,
                                shuffle=True,
                                batch_size=b_size,
                                image_size=img_size)
    return train_ds, validation_ds

def get_base_model(img_shape):

    base_model = \
      EfficientNetV2B0(input_shape=img_shape,
                       include_top=False,
                       weights='imagenet')
    base_model.trainable = False
    return base_model

def predict(model, img_size, path='testdata/test_cat0.jpg'):

    image = load_img(path, target_size=img_size)
    inputs = np.array([img_to_array(image)])
    logits = model.predict(inputs)
    label = tf.nn.sigmoid(logits)[0].numpy()[0]
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    if label < .5:
        plt.title(f'cat w/ probability {(1-label) * 100:.2f}%')
    else:
        plt.title(f'dog w/ probability {label * 100:.2f}%')
    plt.show()

if __name__ == '__main__':

    # set initial data, here channels last
    b_size = 32
    img_size = (160, 160)
    img_shape = img_size + (3,)
    train_model = True

    if train_model: # train the model

        # get training data, look at one batch
        train_ds, validation_ds = get_cats_and_dogs(b_size, img_size)
        image_batch, label_batch = next(iter(train_ds))

        # plot one batch (b_size of 32 is assumed here)
        plt.figure(figsize=(10, 6))
        for k in range(32):
            plt.subplot(4, 8, k+1)
            plt.imshow(image_batch[k].numpy().astype("uint8"))
            plt.title(f'{label_batch[k].numpy().astype(int)}')
            plt.axis('off')
        plt.savefig('cats_and_dogs.png', bbox_inches='tight')
        plt.show()

        # look at output of base model
        base_model = get_base_model(img_shape)
        feature_batch = base_model(image_batch)
        print(feature_batch.shape)

        # define global average pooling layer
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        print(feature_batch_average.shape)

        # define prediction layer
        prediction_layer = tf.keras.layers.Dense(1)
        prediction_batch = prediction_layer(feature_batch_average)
        print(prediction_batch.shape)

        # stitch everything together to obtain the final model
        inputs = tf.keras.Input(shape=(160, 160, 3))
        x = base_model(inputs, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        # look at the resulting model
        print(model.summary())

        # INFO:Uncomment this part if you want to plot the model. This requires
        # two additional python packages: pydot and graphviz.

        # plot_model(model, to_file='transfer_model.pdf',
        #            show_shapes=True)

        # compile the model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=BinaryCrossentropy(from_logits=True),
            metrics=[BinaryAccuracy(threshold=0, name='accuracy')])

        # evaluate the model with the untrained top
        loss0, accuracy0 = model.evaluate(validation_ds)
        print(f'initial loss: {loss0:.2f}')
        print(f'initial accuracy: {accuracy0:.2f}')

        # train the top
        history = model.fit(train_ds,
                            epochs=10,
                            validation_data=validation_ds)

        # extract accuracy and loss
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # plot accuracy and loss
        plt.figure(figsize=(8, 8))
        plt.subplot(211)
        plt.plot(acc, label='training accuracy')
        plt.plot(val_acc, label='validation accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and validation accuracy')
        plt.subplot(212)
        plt.plot(loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and validation loss')
        plt.xlabel('epoch')
        plt.savefig('accuracy_and_loss.pdf', bbox_inches='tight')
        plt.show()

        # evaluate the model with the trained top
        loss1, accuracy1 = model.evaluate(validation_ds)
        print(f'final loss: {loss1:.2f}')
        print(f'final accuracy: {accuracy1:.2f}')

        # save the trained model
        model.save('cats_and_dogs.keras')

    else: # load the trained model

        model = tf.keras.models.load_model('cats_and_dogs.keras')

    # test the model on some images of cats and dogs
    predict(model, img_size, path='testdata/test_cat0.jpg')
    predict(model, img_size, path='testdata/test_dog0.jpg')
    # you might want to add additional ones

    # test the model on some other images
    predict(model, img_size, path='testdata/test_squirrel0.jpg')
    predict(model, img_size, path='testdata/test_human0.jpg')
    # you might want to add additional ones
