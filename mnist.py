import tensorflow as tf
import numpy as np

SHUFFLE_BUFFER_SIZE = 10000


class Mnist:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        self.x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
        self.x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

        self.y_train = np.eye(10)[y_train]  # one hot
        self.y_test = np.eye(10)[y_test]

    def get_train_val_datasets(self, batch_size):
        train_batches = tf.data.Dataset.from_tensor_slices({"features": self.x_train, "label": self.y_train}) \
            .shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        validation_batches = tf.data.Dataset.from_tensor_slices({"features": self.x_test, "label": self.y_test}) \
            .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return train_batches, validation_batches
