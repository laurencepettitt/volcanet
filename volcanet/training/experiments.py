import math

import numpy as np
import tensorflow as tf

import data
from volcanet.training import plotting
from volcanet.training import train

AUTOTUNE = tf.data.experimental.AUTOTUNE
EPOCHS = 100
BATCH_SIZE = 32


def evaluate(model, subset, batch_size, show_first_batch=False):
    """Evaluate model on subset."""
    # Get dataset
    ds, ds_size = data.load_ds(subset, label='class'), data.size(subset, 'class')

    # Prepare dataset for input to model
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
    steps = int(math.floor(ds_size / batch_size))

    # Evaluate model on dataset and print results
    results = model.evaluate(ds, steps=steps)

    # Split dataset into inputs and labels
    inputs_ds = ds.unbatch().map(lambda x, _: x).batch(batch_size)
    labels_ds = ds.unbatch().map(lambda _, y: y)

    # Make predictions on input
    preds = model.predict(inputs_ds, steps=steps)
    # Turn labels into list (same size as preds)
    labels = list(labels_ds.as_numpy_iterator())[:len(preds)]
    return labels, preds, results


def evaluate_both(model):
    """Evaluate the model on both the train and test sets. (Plot confusion matrices and print results)"""
    for subset in ['test', 'train']:
        # Get labels, predictions and results of evaluation
        labels, preds, results = evaluate(model, subset, BATCH_SIZE)
        # Plot confusion matrix
        plotting.plot_cm(labels, np.argmax(preds, axis=1))
        # Print results
        cat_acc = results[1]
        print("{}: {}".format(subset, cat_acc))


def baseline_experiment():
    """The baseline model"""
    model, history = train.train(BATCH_SIZE, EPOCHS, print_model_summary=True)
    evaluate_both(model)
    plotting.plot_metrics(history)


def oversampling_experiment():
    """Oversampling to balance classes"""
    model, history = train.train(BATCH_SIZE, EPOCHS, print_model_summary=True,
                                 oversampling=True)
    evaluate_both(model)
    plotting.plot_metrics(history)


def central_cropping_experiment():
    """Cropping input to a central square"""
    model, history = train.train(BATCH_SIZE, EPOCHS, print_model_summary=True,
                                 central_cropping=True)
    evaluate_both(model)
    plotting.plot_metrics(history)


def cc_os_experiment():
    """Oversampling and cropping_input"""
    model, history = train.train(BATCH_SIZE, EPOCHS, print_model_summary=True,
                                 central_cropping=True, oversampling=True)
    evaluate_both(model)
    plotting.plot_metrics(history)


def evaluate_final():
    model = tf.keras.models.load_model(train.FINAL_MODEL)
    evaluate_both(model)


if __name__ == '__main__':
    # baseline_experiment()
    # oversampling_experiment()
    # central_cropping_experiment()
    # cc_os_experiment()
    evaluate_final()
