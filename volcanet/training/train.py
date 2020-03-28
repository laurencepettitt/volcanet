import math
import os
import pathlib

import sklearn
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import losses, activations, layers, regularizers, models, optimizers, callbacks

import data

SHUFFLE_BUFFER_SIZE = 2000
MODULE_DIR = os.path.dirname(__file__)
FINAL_MODEL = pathlib.Path(os.path.join(MODULE_DIR, 'model-final.h5'))


def show_batch(image_batch, label_batch):
    """Useful to preview a batch of images just before feeding into a model"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for n in range(min(25, len(image_batch))):
        ax = plt.subplot(5, 5, n+1)
        image = np.reshape(image_batch[n], image_batch[n].shape[:2])
        plt.imshow(image, cmap='gray')
        plt.title(str(label_batch[n]))
        plt.axis('off')
    plt.show()


def init_cnn(input_shape, batch_size, num_output_units, output_activation, metrics, loss, central_cropping=False):
    """Build and compile the network."""

    model = models.Sequential(name='volcanet-cnn')

    model.add(layers.Input(input_shape, batch_size))

    if central_cropping:
        model.add(layers.Lambda(lambda x: tf.image.central_crop(x, 0.4)))

    # Convolutional Layers
    model.add(layers.Conv2D(16, kernel_size=(3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, kernel_size=(3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    # Fully-Connected Layers
    model.add(layers.Dense(64))
    model.add(layers.Activation('relu'))

    model.add(layers.Dense(32))
    model.add(layers.Activation('relu'))

    model.add(layers.Dense(num_output_units, name='output'))
    model.add(layers.Activation(output_activation))

    model.compile(
        optimizer=optimizers.Adam(),
        loss=loss,
        metrics=metrics,
    )
    return model


def train_test_split(ds, train_size):
    return ds.take(train_size), ds.skip(train_size)


def prepare_data(oversampling=False):
    """Loads the dataset, splits it into train/val sets and optionally applies oversampling to balance classes"""
    # Load dataset
    ds, ds_size, num_classes = data.load_ds('train'), data.size('train'), data.num_labels('train')

    # Split data into train & val subsets
    val_ds_size = 320
    train_ds_size = ds_size - val_ds_size
    train_ds, val_ds = train_test_split(ds, train_size=train_ds_size)

    # Shuffle and repeat training subset
    train_ds = train_ds.cache().shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=42).repeat()

    # Oversample to balance class counts
    if oversampling:
        class_datasets = [train_ds.filter(lambda x, y: y == c) for c in range(num_classes)]
        train_ds = tf.data.experimental.sample_from_datasets(class_datasets,
                                                             weights=[0.52, 0.04, 0.10, 0.16, 0.18],
                                                             seed=42)

    return train_ds, val_ds, train_ds_size, val_ds_size, num_classes


def prepare_model(num_classes, batch_size, central_cropping=False):
    """Returns a built and compiled model ready for training"""
    loss = losses.sparse_categorical_crossentropy
    metrics = [tf.metrics.SparseCategoricalAccuracy(name='cat_acc')]
    output_activation = activations.softmax
    input_shape = (data.IMG_HEIGHT, data.IMG_WIDTH, data.NUM_CHANNELS)

    return init_cnn(input_shape, batch_size, num_classes, output_activation, metrics, loss,
                    central_cropping=central_cropping)


def train(batch_size, epochs, early_stopping=None, show_first_batch=False, print_model_summary=False,
          oversampling=False, central_cropping=False):
    """Train the classifier under various strategies."""
    # Get training data
    train_ds, val_ds, train_ds_size, val_ds_size, num_classes = prepare_data(oversampling)

    # Build model
    model = prepare_model(num_classes, batch_size, central_cropping=central_cropping)

    # Stop training when validation categorical accuracy stops improving (for many epochs)
    if early_stopping is None:
        early_stopping = callbacks.EarlyStopping(
            monitor='val_cat_acc',
            verbose=1,
            patience=20,
            mode='max',
            restore_best_weights=True)

    # Prepare dataset for training (batching)
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    # Useful for debugging
    if show_first_batch:
        show_batch(*next(train_ds.as_numpy_iterator()))
        show_batch(*next(val_ds.as_numpy_iterator()))

    # Fit data to model
    history = model.fit(train_ds,
                        epochs=epochs,
                        steps_per_epoch=int(math.ceil(train_ds_size / batch_size)),
                        validation_data=val_ds,
                        validation_steps=int(math.ceil(val_ds_size / batch_size)),
                        callbacks=[early_stopping])

    if print_model_summary:
        print(model.summary())

    return model, history


def train_final():
    """Train the classifier under the final training strategy."""
    model, _ = train(32, 100, central_cropping=True)
    model.save(FINAL_MODEL)


if __name__ == '__main__':
    train_final()
