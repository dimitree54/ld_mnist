import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dropout, BatchNormalization, Dense, Flatten, Lambda, LeakyReLU, Reshape
from tensorflow.python.keras.losses import binary_crossentropy


def sampling(args):
    _z_mean, _z_log_var, latent_dim = args
    batch_size = tf.shape(_z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
    return _z_mean + tf.exp(_z_log_var / 2) * epsilon


def calc_reconstruction_loss(input_x, decoded_x):
    reconstruction_loss = binary_crossentropy(input_x, decoded_x)
    return tf.reduce_mean(reconstruction_loss)


def calc_kl_loss(z_mean, z_log_var):
    kl_loss = -tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return kl_loss


def create_image_vae(dropout_rate=0.3, latent_dim=2):
    vae_models = {}

    input_img = Input(shape=(28, 28, 1))
    x = Flatten()(input_img)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    latent_code = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var, latent_dim])

    z = Input(shape=(latent_dim,))
    x = Dense(128)(z)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(28 * 28, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(x)

    vae_models['encoder'] = Model(input_img, latent_code, name='encoder')
    vae_models['decoder'] = Model(z, decoded, name='decoder')
    vae_models['vae'] = Model(input_img, vae_models['decoder'](latent_code), name='vae')

    vae_models["encoder"].add_loss(calc_kl_loss(z_mean, z_log_var))

    return vae_models


def create_middle_layers(dropout_rate=0.3, latent_dim=2):
    z = Input(shape=(latent_dim,))
    x = Dense(latent_dim)(z)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(latent_dim)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(latent_dim)(x)
    return Model(z, x, name="MiddleLayers")


def create_class_vae(latent_dim=2):
    vae_models = {}

    input_class = Input(shape=(10,))
    x = input_class

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    latent_code = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var, latent_dim])

    z = Input(shape=(latent_dim,))
    decoded = Dense(10, activation='softmax')(z)

    vae_models["encoder"] = Model(input_class, latent_code, name='Encoder')
    vae_models["decoder"] = Model(z, decoded, name='Decoder')
    vae_models["vae"] = Model(input_class, vae_models["decoder"](latent_code), name="VAE")

    vae_models["encoder"].add_loss(calc_kl_loss(z_mean, z_log_var))

    return vae_models
