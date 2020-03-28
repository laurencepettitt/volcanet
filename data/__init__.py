import os
import pathlib
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf

MODULE_DIR = os.path.dirname(__file__)

IMG_WIDTH = 110
IMG_HEIGHT = 110
NUM_CHANNELS = 1


def _unzip_image_data(zip_file):
    """
    Unzips png files in zip_files, returning list of their paths
    :param zip_file: path to zip file containing png files
    :return: list of png file paths extracted from zip file
    """
    with zipfile.ZipFile(zip_file, 'r') as z:
        for filepath in z.namelist():
            if filepath.endswith('.png'):
                z.extract(filepath, MODULE_DIR)
                yield os.path.join(MODULE_DIR, filepath)


def _init_dataframe(subset: str):
    """
    Returns dataframe of image paths and their labels.

    Reads image paths from a zip file in this directory (after unzipping)
    and image labels from a csv file in this directory
    """
    assert subset in ['train', 'test']
    images_zip = os.path.join(MODULE_DIR, subset + '_images.zip')  # Original source of images
    labels_csv = os.path.join(MODULE_DIR, subset + '_labels.csv')  # Original source of labels
    # Process labels from csv file into dataframe
    labels_df = pd.read_csv(labels_csv)
    labels_df.columns = ['is_volcano', 'type', 'radius', 'number_volcanoes']
    # We only care about these columns
    labels_df = labels_df[['is_volcano', 'type']]
    labels_df['class'] = labels_df['type'].fillna(0)
    # Process png image filenames found in zip during extraction and add to dataframe in correct order
    filenames = pd.Series(_unzip_image_data(images_zip), name='filename')
    names = pd.Series(filenames.map(lambda p: int(pathlib.Path(p).stem)), name='name', dtype=np.int32)
    # Combine names with filenames and then sort by name (to align them with labels_df)
    files_df = pd.concat([names, filenames], axis=1).sort_values(by='name').reset_index(drop=True)
    # combine filenames with their labels
    full_df = pd.concat([files_df, labels_df], axis=1)
    # Deterministically shuffle data
    return full_df.sample(frac=1, random_state=42)


def _decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=1)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    return tf.image.convert_image_dtype(img, tf.float32)


def process_path(file_path):
    """Loads image from file_path"""
    img = tf.io.read_file(file_path)
    img = _decode_img(img)
    img.set_shape((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
    return img


def load_ds(subset, label='class'):
    """Load data set (drops examples with nan label values)"""
    df = data[subset][['filename'] + [label]].dropna().reset_index(drop=True)

    filenames_list = df['filename'].tolist()
    filename_ds = tf.data.Dataset.from_tensor_slices(filenames_list)
    img_ds = filename_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(df[label].tolist())
    return tf.data.Dataset.zip((img_ds, label_ds))


def size(subset, label='class'):
    """Returns size of the subset with label."""
    return len(data[subset][label].dropna().reset_index(drop=True).index)


def counts(subset, label='class'):
    """Returns the counts (occurrences) of each label in the subset"""
    return data[subset][label].value_counts()


def num_labels(subset, label='class'):
    """Returns the number of different labels in subset"""
    return len(counts(subset, label))


data = {
    'train': _init_dataframe('train'),
    'test': _init_dataframe('test')
}
