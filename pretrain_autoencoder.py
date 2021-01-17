import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import os
import numpy as np

from eval import evaluate
from misc import LRScheduleWithReduceOnPlato
from models import create_image_vae, calc_reconstruction_loss, create_middle_layers, create_class_vae
from plots_drawer import plot_digits, plot_manifold, plot_histogram
import mnist

start_lr = 0.001
batch_size = 10000
n_epochs = 5000
latent_dim = 32
lr_schedule = LRScheduleWithReduceOnPlato(start_lr, 50, 0.9)
tb_dir = "logs_with_class_v3"
saved_models_dir = "saved_models_v3"
simple_dataset_dir = "ld_mnist_dataset_v3"


def train(train_data, val_data, image_vae_models, middle_layers, class_vae_models):
    optimizer = Adam(lr_schedule.get_lr)
    middle_layers_optimizer = Adam(lr_schedule.get_lr)
    all_variables = \
        image_vae_models["encoder"].trainable_variables + image_vae_models["decoder"].trainable_variables + \
        class_vae_models["encoder"].trainable_variables + class_vae_models["decoder"].trainable_variables
    middle_layers_variables = middle_layers.trainable_variables

    # Tensorboard:
    tb_train_writer = tf.summary.create_file_writer(os.path.join(tb_dir, "train"))
    tb_val_writer = tf.summary.create_file_writer(os.path.join(tb_dir, "val"))

    # Metrics:
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

    def calc_outputs(x, y):
        image_latent_code = image_vae_models["encoder"](x)
        image_reconstructed = image_vae_models["decoder"](image_latent_code)
        image2class_latent_code = middle_layers(image_latent_code)
        class_prediction = class_vae_models["decoder"](image2class_latent_code)
        class_latent_code = class_vae_models["encoder"](y)
        class_reconstructed = class_vae_models["decoder"](class_latent_code)
        return image_reconstructed, class_prediction, class_reconstructed

    def calc_losses(x, y, outputs):
        image_reconstructed, class_prediction, class_reconstructed = outputs

        image_vae_internal_loss = tf.reduce_sum(image_vae_models["encoder"].losses) * 0.1
        image_vae_reconstruction_loss = calc_reconstruction_loss(x, image_reconstructed)
        image_vae_total_loss = image_vae_internal_loss + image_vae_reconstruction_loss

        class_vae_internal_loss = tf.reduce_sum(class_vae_models["encoder"].losses) * 0.1
        class_vae_reconstruction_loss = calc_reconstruction_loss(y, class_reconstructed)
        class_vae_total_loss = class_vae_internal_loss + class_vae_reconstruction_loss

        classification_loss = calc_reconstruction_loss(y, class_prediction)

        total_loss = image_vae_total_loss + class_vae_total_loss + classification_loss

        return total_loss, [image_vae_internal_loss, image_vae_reconstruction_loss, image_vae_total_loss,
                            class_vae_internal_loss, class_vae_reconstruction_loss, class_vae_total_loss,
                            classification_loss, total_loss]

    def update_classification_accuracy_metrics(y, outputs):
        image_reconstructed, class_prediction, class_reconstructed = outputs
        class_vae_reconstruction_accuracy_metric.update_state(y, class_reconstructed)
        class_accuracy_metric.update_state(y, class_prediction)

    def update_loss_metrics(losses):
        image_vae_internal_loss, image_vae_reconstruction_loss, image_vae_total_loss, \
            class_vae_internal_loss, class_vae_reconstruction_loss, class_vae_total_loss, \
            classification_loss, total_loss = losses
        image_vae_internal_loss_metric.update_state(image_vae_internal_loss)
        image_vae_reconstruction_loss_metric.update_state(image_vae_reconstruction_loss)
        image_vae_total_loss_metric.update_state(image_vae_total_loss)
        class_vae_internal_loss_metric.update_state(class_vae_internal_loss)
        class_vae_reconstruction_loss_metric.update_state(class_vae_reconstruction_loss)
        class_vae_total_loss_metric.update_state(class_vae_total_loss)
        classification_loss_metric.update_state(classification_loss)
        total_loss_metric.update_state(total_loss)

    def write_metrics():
        tf.summary.scalar("image_vae/internal_loss", image_vae_internal_loss_metric.result(), epoch)
        tf.summary.scalar("image_vae/reconstruction_loss", image_vae_reconstruction_loss_metric.result(), epoch)
        tf.summary.scalar("image_vae/total_loss", image_vae_total_loss_metric.result(), epoch)
        tf.summary.scalar("class_vae/internal_loss", class_vae_internal_loss_metric.result(), epoch)
        tf.summary.scalar("class_vae/reconstruction_loss", class_vae_reconstruction_loss_metric.result(), epoch)
        tf.summary.scalar("class_vae/reconstruction_accuracy", class_vae_reconstruction_accuracy_metric.result(),
                          epoch)
        tf.summary.scalar("class_vae/total_loss", class_vae_total_loss_metric.result(), epoch)
        tf.summary.scalar("class/loss", classification_loss_metric.result(), epoch)
        tf.summary.scalar("class/accuracy", class_accuracy_metric.result(), epoch)
        tf.summary.scalar("total_loss", total_loss_metric.result(), epoch)

    def reset_metrics():
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

    @tf.function
    def test():
        for batch in val_data:
            x = batch["features"]
            y = batch["label"]
            outputs = calc_outputs(x, y)
            _, losses = calc_losses(x, y, outputs)
            update_classification_accuracy_metrics(y, outputs)
            update_loss_metrics(losses)

    @tf.function
    def train_one_epoch():
        for batch in train_data:
            x = batch["features"]
            y = batch["label"]
            with tf.GradientTape(persistent=True) as tape:
                outputs = calc_outputs(x, y)
                total_loss, losses = calc_losses(x, y, outputs)
            grads = tape.gradient(total_loss, all_variables)
            middle_layers_grads = tape.gradient(total_loss, middle_layers_variables)
            optimizer.apply_gradients(zip(grads, all_variables))
            middle_layers_optimizer.apply_gradients(zip(middle_layers_grads, middle_layers_variables))

            update_classification_accuracy_metrics(y, outputs)
            update_loss_metrics(losses)

    for epoch in tqdm(range(n_epochs)):
        train_one_epoch()
        lr_schedule.update_loss(total_loss_metric.result())

        with tb_train_writer.as_default():
            tf.summary.scalar("lr", lr_schedule.get_lr(), epoch)
            write_metrics()
        reset_metrics()

        test()

        with tb_val_writer.as_default():
            write_metrics()
        reset_metrics()


def main():
    image_vae_models = create_image_vae(dropout_rate=0.9, latent_dim=latent_dim)
    middle_layers = create_middle_layers(dropout_rate=0.9, latent_dim=latent_dim)
    class_vae_models = create_class_vae(latent_dim=latent_dim)

    orig_mnist = mnist.Mnist()
    train_data, val_data = orig_mnist.get_train_val_datasets(batch_size)

    train(train_data, val_data, image_vae_models, middle_layers, class_vae_models)

    input("Press ENTER to draw images")
    plot_images(val_data, image_vae_models)

    input("Press ENTER to create simple dataset, save it and show statistics")
    train_x, train_y, test_x, test_y = convert_dataset(orig_mnist, image_vae_models, class_vae_models)

    np.save(os.path.join(os.path.join(simple_dataset_dir, "train_x.npy")), train_x, allow_pickle=False)
    np.save(os.path.join(os.path.join(simple_dataset_dir, "train_y.npy")), train_y, allow_pickle=False)
    np.save(os.path.join(os.path.join(simple_dataset_dir, "test_x.npy")), test_x, allow_pickle=False)
    np.save(os.path.join(os.path.join(simple_dataset_dir, "test_y.npy")), test_y, allow_pickle=False)

    show_dataset_statistics(train_x, train_y)

    input("Press ENTER to save models and eval middle_layers")
    save_models(image_vae_models, class_vae_models)
    evaluate(middle_layers, mnist.LdMnist(simple_dataset_dir), orig_mnist, class_vae_models["decoder"])


def save_models(image_vae_models, class_vae_models):
    if os.path.isdir(saved_models_dir):
        print("WARNING, writing at existing directory")
    else:
        os.mkdir(saved_models_dir)
    image_vae_models["encoder"].save(os.path.join(saved_models_dir, "image_encoder"), include_optimizer=False)
    image_vae_models["decoder"].save(os.path.join(saved_models_dir, "image_decoder"), include_optimizer=False)
    class_vae_models["encoder"].save(os.path.join(saved_models_dir, "class_encoder"), include_optimizer=False)
    class_vae_models["decoder"].save(os.path.join(saved_models_dir, "class_decoder"), include_optimizer=False)


def plot_images(val_data, image_vae_models):
    images = next(iter(val_data))["features"][:10].numpy()
    decoded = image_vae_models['vae'].predict(images)
    plot_digits(images[:10], decoded[:10])
    for i in range(0, latent_dim, 2):
        plot_manifold(image_vae_models["decoder"], latent_dim=latent_dim, x_dim=i, y_dim=i+1)


def convert_dataset(orig_mnist, image_vae_models, class_vae_models):
    train_data, val_data = orig_mnist.get_train_val_datasets(batch_size=batch_size, shuffle=False)
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    if os.path.isdir(simple_dataset_dir):
        print("WARNING, writing at existing directory")
    else:
        os.mkdir(simple_dataset_dir)

    for batch in train_data:
        x = batch["features"]
        y = batch["label"]
        image_latent_code = image_vae_models["encoder"](x)
        class_latent_code = class_vae_models["encoder"](y)
        train_x.append(image_latent_code.numpy())
        train_y.append(class_latent_code.numpy())

    for batch in val_data:
        x = batch["features"]
        y = batch["label"]
        image_latent_code = image_vae_models["encoder"](x)
        class_latent_code = class_vae_models["encoder"](y)
        test_x.append(image_latent_code.numpy())
        test_y.append(class_latent_code.numpy())

    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)

    return train_x, train_y, test_x, test_y


def show_dataset_statistics(train_x, train_y):
    train_x_mean = np.mean(train_x)
    train_x_std = np.std(train_x)
    train_y_mean = np.mean(train_y)
    train_y_std = np.std(train_y)

    print("train_x mean and std:", train_x_mean, train_x_std)
    print("train_y mean and std:", train_y_mean, train_y_std)

    for i in range(latent_dim):
        print(f"train_x mean and std of {i}-th component: {np.mean(train_x[:, i]):.3f} {np.std(train_x[:, i]):.3f}")
        print(f"train_y mean and std of {i}-th component: {np.mean(train_y[:, i]):.3f} {np.std(train_y[:, i]):.3f}")

    plot_histogram(train_x, f"hist of train_x (mean={train_x_mean:.3f}, std={train_x_std:.3f})")
    plot_histogram(train_y, f"hist of train_y (mean={train_y_mean:.3f}, std={train_y_std:.3f})")


if __name__ == "__main__":
    main()
