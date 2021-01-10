import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models import create_vae
from plots_drawer import plot_digits, plot_manifold
import mnist

start_lr = 0.0001
batch_size = 500
n_epochs = 1000


def train(train_data, val_data, vae, vae_loss):
    optimizer = Adam(start_lr)

    tb_train_writer = tf.summary.create_file_writer("logs/train")
    tb_val_writer = tf.summary.create_file_writer("logs/val")

    total_loss_metric = tf.metrics.Mean()
    reconstruction_loss_metric = tf.metrics.Mean()
    internal_loss_metric = tf.metrics.Mean()

    @tf.function
    def train_one_epoch():
        for batch in train_data:
            x_train = batch["features"]
            y_train = batch["label"]
            with tf.GradientTape() as tape:
                reconstructed_x = vae(x_train)
                reconstruction_loss = vae_loss(x_train, reconstructed_x)
                internal_loss = tf.reduce_sum(vae.losses)
                total_loss = reconstruction_loss + internal_loss
            grads = tape.gradient(total_loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(grads, vae.trainable_variables))


            total_loss_metric.update_state(total_loss)
            reconstruction_loss_metric.update_state(reconstruction_loss)
            internal_loss_metric.update_state(internal_loss)
        with tb_train_writer.as_default():
            tf.summary.scalar("total_loss", total_loss_metric.result(), epoch)
            tf.summary.scalar("reconstruction_loss", reconstruction_loss_metric.result(), epoch)
            tf.summary.scalar("internal_loss", internal_loss_metric.result(), epoch)
        total_loss_metric.reset_states()
        reconstruction_loss_metric.reset_states()
        internal_loss_metric.reset_states()

        for batch in val_data:
            x_train = batch["features"]
            y_train = batch["label"]
            reconstructed_x = vae(x_train)
            reconstruction_loss = vae_loss(x_train, reconstructed_x)
            internal_loss = tf.reduce_sum(vae.losses)
            total_loss = reconstruction_loss + internal_loss

            total_loss_metric.update_state(total_loss)
            reconstruction_loss_metric.update_state(reconstruction_loss)
            internal_loss_metric.update_state(internal_loss)
        with tb_val_writer.as_default():
            tf.summary.scalar("total_loss", total_loss_metric.result(), epoch)
            tf.summary.scalar("reconstruction_loss", reconstruction_loss_metric.result(), epoch)
            tf.summary.scalar("internal_loss", internal_loss_metric.result(), epoch)
        total_loss_metric.reset_states()
        reconstruction_loss_metric.reset_states()
        internal_loss_metric.reset_states()

    for epoch in range(n_epochs):
        train_one_epoch()


def main():
    models, vae_loss = create_vae()
    vae = models["vae"]

    train_data, val_data = mnist.get_data(batch_size)

    train(train_data, val_data, vae, vae_loss)

    images = next(iter(val_data))["features"][:10].numpy()
    decoded = vae.predict(images)
    plot_digits(images[:10], decoded[:10])
    plot_manifold(models["decoder"])
    input()


main()
