import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dropout, BatchNormalization, Dense, Flatten, Lambda, LeakyReLU, Reshape
from tensorflow.python.keras.losses import binary_crossentropy


def create_vae(dropout_rate=0.3, latent_dim=2):
    vae_models = {}

    # Добавим Dropout и BatchNormalization
    def apply_bn_and_dropout(input_x):
        return Dropout(dropout_rate)(BatchNormalization()(input_x))

    # Энкодер
    input_img = Input(batch_shape=(28, 28, 1))
    x = Flatten()(input_img)
    x = Dense(256, activation='relu')(x)
    x = apply_bn_and_dropout(x)
    x = Dense(128, activation='relu')(x)
    x = apply_bn_and_dropout(x)

    # Предсказываем параметры распределений
    # Вместо того, чтобы предсказывать стандартное отклонение, предсказываем логарифм вариации
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # Сэмплирование из Q с трюком репараметризации
    def sampling(args):
        _z_mean, _z_log_var = args
        batch_size = tf.shape(_z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return _z_mean + tf.exp(_z_log_var / 2) * epsilon

    latent_code = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    vae_models["encoder"] = Model(input_img, latent_code, 'Encoder')
    vae_models["z_meaner"] = Model(input_img, z_mean, 'Enc_z_mean')
    vae_models["z_lvarer"] = Model(input_img, z_log_var, 'Enc_z_log_var')

    # Декодер
    z = Input(shape=(latent_dim,))
    x = Dense(128)(z)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(28 * 28, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(x)

    vae_models["decoder"] = Model(z, decoded, name='Decoder')
    vae_models["vae"] = Model(input_img, vae_models["decoder"](latent_code), name="VAE")

    def calc_vae_loss(input_x, decoded_x):
        input_x = tf.reshape(input_x, shape=(-1, 28 * 28, 1))
        decoded_x = tf.reshape(decoded_x, shape=(-1, 28 * 28, 1))
        reconstruction_loss = 28 * 28 * binary_crossentropy(input_x, decoded_x)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        return (reconstruction_loss + kl_loss) / 2 / 28 / 28

    return vae_models, calc_vae_loss
