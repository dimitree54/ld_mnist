import tensorflow as tf

from mnist import Mnist

original_mnist = Mnist()


def evaluate(test_x, test_y, model_for_eval, class_decoder):
    pred_y = model_for_eval(test_x)
    reconstructed_class = class_decoder(pred_y)
    mse = tf.reduce_mean(tf.keras.losses.mse(test_y, pred_y))
    accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(original_mnist.y_test, reconstructed_class))
    print(f"mse: {mse}, reconstructed accuracy: {accuracy}")
