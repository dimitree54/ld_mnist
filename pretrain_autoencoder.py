import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import os

from misc import LRScheduleWithReduceOnPlato
from models import create_image_vae, calc_reconstruction_loss, create_middle_layers, create_class_vae
from plots_drawer import plot_digits, plot_manifold
import mnist

start_lr = 0.001
batch_size = 500
n_epochs = 1000
latent_dim = 128

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    
    
def train_with_class(train_data, val_data, image_vae_models, middle_layers, class_vae_models):
    lr_schedule = LRScheduleWithReduceOnPlato(start_lr, 25, 0.9)
    optimizer = Adam(lr_schedule.get_lr)

    tb_train_writer = tf.summary.create_file_writer("logs_with_class_only_class/train")
    tb_val_writer = tf.summary.create_file_writer("logs_with_class_only_class/val")

    image_vae_reconstruction_loss_metric = tf.metrics.Mean()
    image_vae_internal_loss_metric = tf.metrics.Mean()
    image_vae_total_loss_metric = tf.metrics.Mean()

    class_vae_reconstruction_loss_metric = tf.metrics.Mean()
    class_vae_reconstruction_accuracy_metric = tf.metrics.CategoricalAccuracy()
    class_vae_internal_loss_metric = tf.metrics.Mean()
    class_vae_total_loss_metric = tf.metrics.Mean()

    classification_loss_metric = tf.metrics.Mean()
    class_accuracy_metric = tf.metrics.CategoricalAccuracy()

    total_loss_metric = tf.metrics.Mean()

    all_variables = class_vae_models["encoder"].trainable_variables + class_vae_models["decoder"].trainable_variables #+\
        #image_vae_models["encoder"].trainable_variables + image_vae_models["decoder"].trainable_variables +\
        #middle_layers.trainable_variables

    @tf.function
    def test():
        for batch in val_data:
            x_train = batch["features"]
            y_train = batch["label"]
            #image_latent_code = image_vae_models["encoder"](x_train)
            #image_reconstructed = image_vae_models["decoder"](image_latent_code)
            #image2class_latent_code = middle_layers(image_latent_code)
            #class_prediction = class_vae_models["decoder"](image2class_latent_code)
            class_latent_code = class_vae_models["encoder"](y_train)
            class_reconstructed = class_vae_models["decoder"](class_latent_code)

            #image_vae_internal_loss = tf.reduce_sum(image_vae_models["encoder"].losses)
            #image_vae_reconstruction_loss = calc_reconstruction_loss(x_train, image_reconstructed)
            #image_vae_total_loss = image_vae_internal_loss + image_vae_reconstruction_loss

            class_vae_internal_loss = tf.reduce_sum(class_vae_models["encoder"].losses)
            class_vae_reconstruction_loss = calc_reconstruction_loss(y_train, class_reconstructed)
            class_vae_total_loss = class_vae_reconstruction_loss

            #classification_loss = calc_reconstruction_loss(y_train, class_prediction)

            total_loss = class_vae_total_loss#image_vae_total_loss + class_vae_total_loss + classification_loss

            #image_vae_internal_loss_metric.update_state(image_vae_internal_loss)
            #image_vae_reconstruction_loss_metric.update_state(image_vae_reconstruction_loss)
            #image_vae_total_loss_metric.update_state(image_vae_total_loss)
            class_vae_internal_loss_metric.update_state(class_vae_internal_loss)
            class_vae_reconstruction_loss_metric.update_state(class_vae_reconstruction_loss)
            class_vae_reconstruction_accuracy_metric.update_state(y_train, class_reconstructed)
            class_vae_total_loss_metric.update_state(class_vae_total_loss)
            #classification_loss_metric.update_state(classification_loss)
            #class_accuracy_metric.update_state(y_train, class_prediction)
            total_loss_metric.update_state(total_loss)

    @tf.function
    def train_one_epoch():
        for batch in train_data:
            x_train = batch["features"]
            y_train = batch["label"]
            with tf.GradientTape() as tape:
                #image_latent_code = image_vae_models["encoder"](x_train)
                #image_reconstructed = image_vae_models["decoder"](image_latent_code)
                #image2class_latent_code = middle_layers(image_latent_code)
                #class_prediction = class_vae_models["decoder"](image2class_latent_code)
                class_latent_code = class_vae_models["encoder"](y_train)
                class_reconstructed = class_vae_models["decoder"](class_latent_code)

                #image_vae_internal_loss = tf.reduce_sum(image_vae_models["encoder"].losses)
                #image_vae_reconstruction_loss = calc_reconstruction_loss(x_train, image_reconstructed)
                #image_vae_total_loss = image_vae_internal_loss + image_vae_reconstruction_loss

                class_vae_internal_loss = tf.reduce_sum(class_vae_models["encoder"].losses)
                class_vae_reconstruction_loss = calc_reconstruction_loss(y_train, class_reconstructed)
                class_vae_total_loss = class_vae_internal_loss + class_vae_reconstruction_loss

                #classification_loss = calc_reconstruction_loss(y_train, class_prediction)

                total_loss = class_vae_total_loss#image_vae_total_loss + class_vae_total_loss + classification_loss
            grads = tape.gradient(total_loss, all_variables)
            optimizer.apply_gradients(zip(grads, all_variables))

            #image_vae_internal_loss_metric.update_state(image_vae_internal_loss)
            #image_vae_reconstruction_loss_metric.update_state(image_vae_reconstruction_loss)
            #image_vae_total_loss_metric.update_state(image_vae_total_loss)
            class_vae_internal_loss_metric.update_state(class_vae_internal_loss)
            class_vae_reconstruction_loss_metric.update_state(class_vae_reconstruction_loss)
            class_vae_reconstruction_accuracy_metric.update_state(y_train, class_reconstructed)
            class_vae_total_loss_metric.update_state(class_vae_total_loss)
            #classification_loss_metric.update_state(classification_loss)
            #class_accuracy_metric.update_state(y_train, class_prediction)
            total_loss_metric.update_state(total_loss)

    for epoch in tqdm(range(n_epochs)):
        train_one_epoch()
        lr_schedule.update_loss(total_loss_metric.result())

        with tb_train_writer.as_default():
            tf.summary.scalar("lr", lr_schedule.get_lr(), epoch)
            tf.summary.scalar("image_vae/internal_loss", image_vae_internal_loss_metric.result(), epoch)
            tf.summary.scalar("image_vae/reconstruction_loss", image_vae_reconstruction_loss_metric.result(), epoch)
            tf.summary.scalar("image_vae/total_loss", image_vae_total_loss_metric.result(), epoch)
            tf.summary.scalar("class_vae/internal_loss", class_vae_internal_loss_metric.result(), epoch)
            tf.summary.scalar("class_vae/reconstruction_loss", class_vae_reconstruction_loss_metric.result(), epoch)
            tf.summary.scalar("class_vae/reconstruction_accuracy", class_vae_reconstruction_accuracy_metric.result(), epoch)
            tf.summary.scalar("class_vae/total_loss", class_vae_total_loss_metric.result(), epoch)
            tf.summary.scalar("class/loss", classification_loss_metric.result(), epoch)
            tf.summary.scalar("class/accuracy", class_accuracy_metric.result(), epoch)
            tf.summary.scalar("total_loss", total_loss_metric.result(), epoch)
        image_vae_internal_loss_metric.reset_states()
        image_vae_reconstruction_loss_metric.reset_states()
        image_vae_total_loss_metric.reset_states()
        class_vae_internal_loss_metric.reset_states()
        class_vae_reconstruction_loss_metric.reset_states()
        class_vae_reconstruction_accuracy_metric.reset_states()
        class_vae_total_loss_metric.reset_states()
        classification_loss_metric.reset_states()
        class_accuracy_metric.reset_states()
        total_loss_metric.reset_states()

        test()

        print(class_vae_reconstruction_accuracy_metric.result())

        with tb_val_writer.as_default():
            tf.summary.scalar("image_vae/internal_loss", image_vae_internal_loss_metric.result(), epoch)
            tf.summary.scalar("image_vae/reconstruction_loss", image_vae_reconstruction_loss_metric.result(), epoch)
            tf.summary.scalar("image_vae/total_loss", image_vae_total_loss_metric.result(), epoch)
            tf.summary.scalar("class_vae/internal_loss", class_vae_internal_loss_metric.result(), epoch)
            tf.summary.scalar("class_vae/reconstruction_loss", class_vae_reconstruction_loss_metric.result(), epoch)
            tf.summary.scalar("class_vae/reconstruction_accuracy", class_vae_reconstruction_accuracy_metric.result(), epoch)
            tf.summary.scalar("class_vae/total_loss", class_vae_total_loss_metric.result(), epoch)
            tf.summary.scalar("class/loss", classification_loss_metric.result(), epoch)
            tf.summary.scalar("class/accuracy", class_accuracy_metric.result(), epoch)
            tf.summary.scalar("total_loss", total_loss_metric.result(), epoch)
        image_vae_internal_loss_metric.reset_states()
        image_vae_reconstruction_loss_metric.reset_states()
        image_vae_total_loss_metric.reset_states()
        class_vae_internal_loss_metric.reset_states()
        class_vae_reconstruction_loss_metric.reset_states()
        class_vae_reconstruction_accuracy_metric.reset_states()
        class_vae_total_loss_metric.reset_states()
        classification_loss_metric.reset_states()
        class_accuracy_metric.reset_states()
        total_loss_metric.reset_states()


def main_with_class():
    image_vae_models = create_image_vae(latent_dim=latent_dim)
    middle_layers = create_middle_layers(latent_dim=latent_dim)
    class_vae_models = create_class_vae(latent_dim=latent_dim)

    train_data, val_data = mnist.get_data(batch_size)

    train_with_class(train_data, val_data, image_vae_models, middle_layers, class_vae_models)

    images = next(iter(val_data))["features"][:10].numpy()
    decoded = image_vae_models['vae'].predict(images)
    plot_digits(images[:10], decoded[:10])
    plot_manifold(image_vae_models["decoder"], latent_dim=latent_dim)
    input()

    if not os.path.isdir("saved_model"):
        os.mkdir("saved_model")
    image_vae_models["encoder"].save("saved_model/image_encoder")
    image_vae_models["decoder"].save("saved_model/image_decoder")
    middle_layers.save("saved_model/middle_layers")
    class_vae_models["encoder"].save("saved_model/class_encoder")
    class_vae_models["decoder"].save("saved_model/class_decoder")


main_with_class()
