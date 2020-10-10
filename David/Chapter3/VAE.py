from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model

from keras.datasets import mnist
import numpy as np

class Vae:

    def __init__(self, img_shape, z_dim):

        self.r_loss_factor = 1000

        optimizer = Adam()


        self.model = self.build_vae(img_shape, z_dim)
        self.model.summary()
        self.model.compile(optimizer=optimizer, loss=self.vae_loss, metrics = [self.r_loss, self.kl_loss])


    def build_encoder(self, img_shape, z_dim):

        input_layer = Input(shape=img_shape)

        x = Conv2D(32, kernel_size=3, strides=1, padding='same')(input_layer)
        x = LeakyReLU()(x)
        
        x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU()(x)

        x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU()(x)

        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        # 分布の平均
        self.mu = Dense(z_dim)(x)
        # 各次元の分散の対数
        self.log_var = Dense(z_dim)(x)

        def sampling(args):
                mu, log_var = args
                epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
                return mu + K.exp(log_var / 2) * epsilon

        output_layer = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        return Model(input_layer, output_layer)

    def build_decoder(self, z_dim):

        # (2)
        input_layer = Input(shape=(z_dim,))

        # (2) => (3136) => (7, 7, 64)
        x = Dense(3136)(input_layer)
        x = Reshape((7, 7, 64))(x)

        # (7, 7, 64) => (7, 7, 64)
        x = Conv2DTranspose(64, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU()(x)

        # (7, 7, 64) => (14, 14, 64)
        x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        
        # (14, 14, 64) => (28, 28, 64)
        x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
        x = LeakyReLU()(x)

        # (28, 28, 64) => (28, 28, 1)
        x = Conv2DTranspose(1, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU()(x)

        output_layer = Activation('tanh')(x)

        return Model(input_layer, output_layer)

    def build_vae(self, img_shape, z_dim):
        encoder = self.build_encoder(img_shape, z_dim)
        decoder = self.build_decoder(z_dim)

        model = Sequential()
        model.add(encoder)
        model.add(decoder)

        return model


    def r_loss(self, y_true, y_pred):
        r_loss = K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])
        return self.r_loss_factor * r_loss

    def kl_loss(self, y_true, y_pred):
        kl_loss = -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis=1)
        return kl_loss

    def vae_loss(self, y_true, y_pred):
        r_loss = self.r_loss(y_true, y_pred)
        kl_loss = self.kl_loss(y_true, y_pred)
        return r_loss + kl_loss

    def train(self, x, y, batch_size, epochs):
        self.model.fit(x, y, batch_size=batch_size, shuffle=True, epochs=epochs)


K.clear_session()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 127.5 - 1.0
x_test = x_test / 127.5 - 1.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
print(x_train.shape)

img_shape = (28, 28, 1)
z_dim = 2
batch_size = 64
epochs = 10

vae = Vae(img_shape, z_dim)
vae.train(x_train, x_train, batch_size, epochs)