import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dropout, BatchNormalization, Dense, Flatten, Lambda, LeakyReLU, Reshape, \
    ReLU, MaxPool2D, Conv2D, UpSampling2D
from tensorflow.python.keras.losses import binary_crossentropy


def sampling(args):
    _z_mean, _z_log_var, latent_dim = args
    batch_size = tf.shape(_z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
    return _z_mean + tf.exp(_z_log_var / 2) * epsilon


def calc_reconstruction_loss(input_x, decoded_x):
    reconstruction_loss = 28 * 28 * binary_crossentropy(input_x, decoded_x) / 2 / 28 / 28
    return tf.reduce_mean(reconstruction_loss)


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

    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1) / 2 / 28 / 28
    vae_models["vae"].add_loss(tf.reduce_mean(kl_loss))

    return vae_models


def create_cnn_vae(dropout_rate=0.3, latent_dim=2):
    vae_models = {}

    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(2, 3, use_bias=False, padding='same')(input_img)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPool2D()(x)
    x = Conv2D(4, 3, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)  # 7*7*4=196
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
    x = Dense(196)(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Reshape((7, 7, 4))(x)
    x = UpSampling2D()(x)
    x = Conv2D(2, 3, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling2D()(x)
    x = Conv2D(1, 3, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    decoded = Conv2D(1, 1, activation='sigmoid')(x)

    vae_models['encoder'] = Model(input_img, latent_code, name='encoder')
    vae_models['decoder'] = Model(z, decoded, name='decoder')
    vae_models['vae'] = Model(input_img, vae_models['decoder'](latent_code), name='vae')

    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1) / 2 / 28 / 28
    vae_models["vae"].add_loss(tf.reduce_mean(kl_loss))

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


def create_class_vae(dropout_rate=0.3, latent_dim=2):
    vae_models = {}

    input_class = Input(shape=(10,))
    x = input_class

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    latent_code = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var, latent_dim])

    z = Input(shape=(latent_dim,))
    decoded = Dense(latent_dim, activation='softmax')(z)

    vae_models["encoder"] = Model(input_class, latent_code, name='Encoder')
    vae_models["decoder"] = Model(z, decoded, name='Decoder')
    vae_models["vae"] = Model(input_class, vae_models["decoder"](latent_code), name="VAE")

    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_models["vae"].add_loss(tf.reduce_sum(kl_loss))

    return vae_models
