import os
import pathlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from sys import argv

from volcanet import predict_ds
import data


def main():
    """Load images from directory and print predictions"""
    # Get data from directory specified by argument one
    data_dir = argv[1]
    data_dir = pathlib.Path(data_dir)
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*.png'), shuffle=False)
    img_ds = list_ds.map(data.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Make predictions
    predictions = predict_ds(img_ds)

    # Print with index
    for i, p in enumerate(predictions):
        print("{}: {}".format(i, p))


if __name__ == '__main__':
    main()
