import tensorflow as tf
import os

from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from mnist import Mnist, LdMnist
from models import create_middle_layers


def evaluate(model_for_eval, ld_mnist, original_mnist, class_decoder):
    pred_y = model_for_eval(ld_mnist.test_x)
    reconstructed_class = class_decoder(pred_y)
    mse = tf.reduce_mean(tf.keras.losses.mse(ld_mnist.test_y, pred_y))
    accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(original_mnist.test_y, reconstructed_class))
    print(f"mse: {mse}, reconstructed accuracy: {accuracy}")


def create_and_train_linear_model(latent_dim, ld_mnist):
    x = Input(shape=(latent_dim,))
    y = Dense(latent_dim)(x)
    model = Model(x, y)
    model.compile(Adam(0.001), MeanSquaredError())
    model.fit(ld_mnist.train_x, ld_mnist.train_y, batch_size=10000, epochs=5000, verbose=0,
              callbacks=ReduceLROnPlateau('loss', 0.9, 50))
    return model


def create_and_train_middle_layers(latent_dim, ld_mnist):
    model = create_middle_layers(0.1, latent_dim)
    model.compile(Adam(0.001), MeanSquaredError())
    model.fit(ld_mnist.train_x, ld_mnist.train_y, batch_size=10000, epochs=5000, verbose=0,
              callbacks=ReduceLROnPlateau('loss', 0.9, 50))
    return model


def main():
    latent_dim = 32
    original_mnist = Mnist()
    ld_mnist = LdMnist("ld_mnist_dataset_v3")
    class_decoder = tf.keras.models.load_model(os.path.join("saved_models_v3", "class_decoder"))

    linear_model = create_and_train_linear_model(latent_dim, ld_mnist)
    print("Linear model evaluation:")
    evaluate(linear_model, ld_mnist, original_mnist, class_decoder)
    del linear_model

    middle_layers_model = create_and_train_middle_layers(latent_dim, ld_mnist)
    print("Middle layers trained from scratch evaluation:")
    evaluate(middle_layers_model, ld_mnist, original_mnist, class_decoder)
    del middle_layers_model


if __name__ == "__main__":
    main()
