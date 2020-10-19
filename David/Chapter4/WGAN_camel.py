from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout, UpSampling2D
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam
import numpy as np

class GAN:

    def __init__(self, img_shape, z_dim, clip_threshold=0.01):

        self.img_shape = img_shape
        self.z_dim = z_dim
        self.clip_threshold = clip_threshold

        self.optimizer_d = Adam(lr=0.00005, beta_1=0.5)
        self.optimizer_g = Adam(lr=0.00005, beta_1=0.5)


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

        # (14, 14, 128) => (28, 28, 64)
        x = UpSampling2D()(x)
        x = Conv2D(64, kernel_size=5, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        # (28, 28, 64) => (28, 28, 64)
        x = Conv2D(64, kernel_size=5, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        #(28, 28, 64) => (28, 28, 1) 
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

        output_layer = Dense(1)(x)

        return Model(input_layer, output_layer)

    def build_GAN(self):

        model = Sequential()

        model.add(self.generator)
        model.add(self.critic)

        return model

    def compile(self):

        def wasserstein(y_true, y_pred):
            return -K.mean(y_true * y_pred)


        # 識別器を訓練するモデルのコンパイル
        self.critic = self.build_discriminator()
        self.critic.compile(optimizer=self.optimizer_d, loss=wasserstein)

        # 生成器を訓練するモデルのコンパイル
        self.critic.trainable = False
        self.generator = self.build_generator()
        self.gan = self.build_GAN()
        self.gan.compile(optimizer=self.optimizer_g, loss=wasserstein)

    def train_discriminator(self, x_train, batch_size):

        valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))

        # 実際の画像で訓練
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        valid_imgs = x_train[idx]
        self.critic.train_on_batch(valid_imgs, valid)

        # 生成された画像で訓練
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = self.generator.predict(noise)
        self.critic.train_on_batch(gen_imgs, fake)

        for l in self.critic.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -self.clip_threshold, self.clip_threshold) for w in weights]
            l.set_weights(weights)

    def train_generator(self, batch_size):
        
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        self.gan.train_on_batch(noise, valid)

    def train_on_batch(self, x_train, batch_size, epochs):
        
        for epoch in range(epochs):
            print('\repoch : %d' %(epoch), end='')
            self.train_discriminator(x_train, batch_size)
            self.train_generator(batch_size)


row = 28
col = 28
channel = 1
img_shape = (row, col, channel)
batch_size = 64
epochs = 1000
z_dim = 100

# (121399, 784) => (121399, 28, 28, 1)
dataset = np.load('dataset/full_numpy_bitmap_camel.npy')
dataset = np.reshape(dataset, (dataset.shape[0], row, col, channel))
dataset = dataset / 127.5 - 1.0

gan = GAN(img_shape, z_dim)
gan.compile()
gan.train_on_batch(dataset, batch_size, epochs)
