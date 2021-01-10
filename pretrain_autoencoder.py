import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models import create_vae
from plots_drawer import plot_digits, plot_manifold

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

models, vae_loss = create_vae()
vae = models["vae"]

start_lr = 0.0001
batch_size = 500
vae.compile(optimizer=Adam(start_lr), loss=vae_loss)

# Коллбэки
lr_red = ReduceLROnPlateau(factor=0.1, patience=25)
tb = TensorBoard(log_dir='./logs')

# Запуск обучения
vae.fit(x_train, x_train, shuffle=True, epochs=1000,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[lr_red, tb],
        verbose=1)

imgs = x_test[:10]
decoded = vae.predict(imgs)
plot_digits(imgs[:10], decoded[:10])
plot_manifold(models["decoder"])
