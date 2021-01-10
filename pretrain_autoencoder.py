import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from models import create_image_vae, calc_reconstruction_loss
from plots_drawer import plot_digits, plot_manifold
import mnist

start_lr = 0.0001
batch_size = 500
n_epochs = 1000

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)


class LRScheduleWithReduceOnPlato:
    def __init__(self, init_lr, patience, factor):
        self.lr = init_lr
        self.last_loss = float("inf")
        self.counter = 0
        self.patience = patience
        self.factor = factor

    def get_lr(self):
        return self.lr

    def update_loss(self, cur_loss):
        if cur_loss >= self.last_loss:
            self.counter += 1
            if self.counter == self.patience:
                self.counter = 0
                self.lr = self.lr * self.factor
        self.last_loss = min(self.last_loss, cur_loss)

    @staticmethod
    def round(x, decimals=0):
        multiplier = tf.constant(10 ** decimals, dtype=tf.float32)
        return tf.round(x * multiplier) / multiplier


def train(train_data, val_data, vae):
    lr_schedule = LRScheduleWithReduceOnPlato(start_lr, 25, 0.1)
    optimizer = Adam(lr_schedule.get_lr)

    tb_train_writer = tf.summary.create_file_writer("logs/train")
    tb_val_writer = tf.summary.create_file_writer("logs/val")

    total_loss_metric = tf.metrics.Mean()
    reconstruction_loss_metric = tf.metrics.Mean()
    internal_loss_metric = tf.metrics.Mean()

    @tf.function
    def test():
        for batch in val_data:
            x_train = batch["features"]
            y_train = batch["label"]
            reconstructed_x = vae(x_train)
            reconstruction_loss = calc_reconstruction_loss(x_train, reconstructed_x)
            internal_loss = tf.reduce_sum(vae.losses)
            total_loss = reconstruction_loss + internal_loss

            total_loss_metric.update_state(total_loss)
            reconstruction_loss_metric.update_state(reconstruction_loss)
            internal_loss_metric.update_state(internal_loss)

    @tf.function
    def train_one_epoch():
        for batch in train_data:
            x_train = batch["features"]
            y_train = batch["label"]
            with tf.GradientTape() as tape:
                reconstructed_x = vae(x_train)
                reconstruction_loss = calc_reconstruction_loss(x_train, reconstructed_x)
                internal_loss = tf.reduce_sum(vae.losses)
                total_loss = reconstruction_loss + internal_loss
            grads = tape.gradient(total_loss, vae.trainable_variables)
            optimizer.apply_gradients(zip(grads, vae.trainable_variables))

            total_loss_metric.update_state(total_loss)
            reconstruction_loss_metric.update_state(reconstruction_loss)
            internal_loss_metric.update_state(internal_loss)

    for epoch in tqdm(range(n_epochs)):
        train_one_epoch()
        lr_schedule.update_loss(total_loss_metric.result())

        with tb_train_writer.as_default():
            tf.summary.scalar("lr", lr_schedule.get_lr(), epoch)
            tf.summary.scalar("total_loss", total_loss_metric.result(), epoch)
            tf.summary.scalar("reconstruction_loss", reconstruction_loss_metric.result(), epoch)
            tf.summary.scalar("internal_loss", internal_loss_metric.result(), epoch)
        total_loss_metric.reset_states()
        reconstruction_loss_metric.reset_states()
        internal_loss_metric.reset_states()

        test()

        with tb_val_writer.as_default():
            tf.summary.scalar("total_loss", total_loss_metric.result(), epoch)
            tf.summary.scalar("reconstruction_loss", reconstruction_loss_metric.result(), epoch)
            tf.summary.scalar("internal_loss", internal_loss_metric.result(), epoch)
        total_loss_metric.reset_states()
        reconstruction_loss_metric.reset_states()
        internal_loss_metric.reset_states()


def main():
    models = create_image_vae()
    vae = models["vae"]

    train_data, val_data = mnist.get_data(batch_size)

    train(train_data, val_data, vae)

    images = next(iter(val_data))["features"][:10].numpy()
    decoded = vae.predict(images)
    plot_digits(images[:10], decoded[:10])
    plot_manifold(models["decoder"])
    input()


main()
