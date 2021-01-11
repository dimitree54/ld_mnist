import tensorflow as tf
import numpy as np

SHUFFLE_BUFFER_SIZE = 10000


def get_data(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    y_train = np.eye(10)[y_train]  # one hot
    y_test = np.eye(10)[y_test]

    train_batches = tf.data.Dataset.from_tensor_slices({"features": x_train, "label": y_train}) \
        .shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    validation_batches = tf.data.Dataset.from_tensor_slices({"features": x_test, "label": y_test}) \
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_batches, validation_batches
