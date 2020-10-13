from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout, UpSampling2D
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam
import numpy as np

class GAN:

    def __init__(self, img_shape, z_dim):

        self.img_shape = img_shape
        self.z_dim = z_dim

        self.potimizer = Adam(lr=0.0002, beta_1=0.5)


    def build_generator(self):
        
        # (100)
        input_layer = Input(shape=(self.z_dim,))

        # (100) => (7, 7, 64)
        x = Dense(3136)(input_layer)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        x = Reshape((7, 7, 64))(x)

        # (7, 7, 64) => (14, 14, 128)
        x = UpSampling2D()(x)
        x = Conv2D(128, kernel_size=5, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=5, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = Conv2D(64, kernel_size=5, strides=1, padding='same')(x)

        x = UpSampling2D()(x)
        x = Conv2D(1, kernel_size=5, strides=1, padding='same')(x)

        output_layer = Activation('tanh')(x)

        return Model(input_layer, output_layer)


    def build_discriminator(self):
        
        # (28, 28, 1)
        input_layer = Input(shape=self.img_shape)

        # (28, 28, 1) => (14, 14, 64)
        x = Conv2D(64, kernel_size=5, strides=2, padding='same')(input_layer)
        x = Activation('relu')(x)
        x = Dropout(rate=0.4)(x)

        # (14, 14, 64) => (7, 7, 64)
        x = Conv2D(64, kernel_size=5, strides=2, padding='same')(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.4)(x)

        # (7, 7, 64) => (4, 4, 128)
        x = Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.4)(x)

        # (4, 4, 128) => (4, 4, 128)
        x = Conv2D(128, kernel_size=5, strides=1, padding='same')(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.4)(x)

        x = Flatten()(x)

        output_layer = Dense(1, activation='sigmoid')(x)

        return Model(input_layer, output_layer)

    def build_GAN(self, generator, discriminator):

        input_layer = Input(shape=(self.z_dim,))

        x = generator(input_layer)
        output_layer = discriminator(x)

        return Model(input_layer, output_layer)

    def compile(self):

        # 識別器を訓練するモデルのコンパイル
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # 生成器を訓練するモデルのコンパイル
        self.discriminator.trainable = False
        self.generator = self.build_generator()
        self.gan = self.build_GAN(self.generator, self.build_discriminator)
        self.gan.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'], experimental_run_tf_function=False)

