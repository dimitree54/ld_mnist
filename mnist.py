import tensorflow as tf
import numpy as np
import os

SHUFFLE_BUFFER_SIZE = 10000


class Mnist:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        self.train_x = np.reshape(x_train, (len(x_train), 28, 28, 1))
        self.test_x = np.reshape(x_test, (len(x_test), 28, 28, 1))

        self.train_y = np.eye(10)[y_train]  # one hot
        self.test_y = np.eye(10)[y_test]

    def get_train_val_datasets(self, batch_size, shuffle=True):
        train_batches = tf.data.Dataset.from_tensor_slices({"features": self.train_x, "label": self.train_y})
        if shuffle:
            train_batches = train_batches.shuffle(SHUFFLE_BUFFER_SIZE)
        train_batches = train_batches.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        validation_batches = tf.data.Dataset.from_tensor_slices({"features": self.test_x, "label": self.test_y}) \
            .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return train_batches, validation_batches


class LdMnist:
    def __init__(self, dataset_dir):
        self.train_x = np.load(os.path.join(os.path.join(dataset_dir, "train_x.npy")))
        self.train_y = np.load(os.path.join(os.path.join(dataset_dir, "train_y.npy")))
        self.test_x = np.load(os.path.join(os.path.join(dataset_dir, "test_x.npy")))
        self.test_y = np.load(os.path.join(os.path.join(dataset_dir, "test_y.npy")))

    def get_train_val_datasets(self, batch_size):
        train_batches = tf.data.Dataset.from_tensor_slices({"features": self.train_x, "label": self.train_y}) \
            .shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        validation_batches = tf.data.Dataset.from_tensor_slices({"features": self.test_x, "label": self.test_y}) \
            .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return train_batches, validation_batches
