import os
import tensorflow as tf
import numpy as np

from volcanet.training.train import FINAL_MODEL as CLASSIFIER

MODULE_DIR = os.path.dirname(__file__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def predict_ds(img_ds):
    """Returns predictions from a dataset of images"""
    img_count = len(list(img_ds.as_numpy_iterator()))

    # Load classifier
    classifier = tf.keras.models.load_model(str(CLASSIFIER))

    # Predict classes
    preds_raw = classifier.predict(img_ds.batch(1).prefetch(buffer_size=500), steps=img_count)

    # Process predictions
    preds = [np.argmax(p) for p in preds_raw]

    return preds
