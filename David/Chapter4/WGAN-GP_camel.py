from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout, UpSampling2D
from keras.layers.merge import _Merge
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
from functools import partial

# 本物の画像のバッチと偽物の画像のバッチの間の直線のランダムな位置にある画像(補間画像)を取得
class RandomWeightedAverage(_Merge):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
    def _merge_function(self, inputs):
        # alpha : 各画像の本物と偽物の割合
        alpha = K.random_uniform((self.batch_size, 0, 1, 1))
        # ピクセルごとに保管画像の集合を返す  inputs[0]:本物の画像,inputs[1]:偽物の画像
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class GAN:

    def __init__(self, img_shape, z_dim, batch_size, clip_threshold=0.01):

        self.img_shape = img_shape
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.clip_threshold = clip_threshold

        self.grad_weight = 10

        self.optimizer_c = Adam(lr=0.00005, beta_1=0.5)
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

        def gradient_penalty_loss(y_true, y_pred, interpolated_samples):

            # 補間画像に対しての予測値(y_pred)の勾配の計算
            gradients = K.gradients(y_pred, interpolated_samples)[0]

            # gradientsベクトルのL2ノルム(ユークリッド距離)の計算
            gradient_l2_norm = K.sqrt(K.sum(K.square(gradients), axis=[1:len(gradients.shape)]))
            
            # L2ノルムと1の間の距離の二乗を返す
            gradient_penalty = K.square(1 - gradient_l2_norm)
            return K.mean(gradient_penalty)


        """ 識別器を訓練するモデルのコンパイル"""

        self.generator = self.build_generator()
        self.critic = self.build_discriminator()

        # 生成器が評価機を訓練するモデルに必要なため、生成器の重みを凍結
        self.set_trainable(self.generator, False)

        # 本物の画像のバッチとランダムな値zの二つを入力にとる
        real_img = Input(shape=self.img_shape)
        z_disc = Input(shape=(self.z_dim,))
        fake_img = self.generator(z_disc)

        # 本物と偽の画像はwasserstein損失を計算するために評価機に渡される
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # 補間画像作成
        interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img])
        # 補間画像はgradient_penalty損失を計算するために評価機に渡される
        validity_interpolated = self.critic(interpolated_img)

        # kerasは損失関数に予測値と正解ラベルの二つしか渡せないため、partialを使いpartial_gp_lossを定義し、補間画像をgradient_penalty__lossに渡している
        partial_gp_loss = partial(gradient_penalty_loss, interpolated_samples = interpolated_img)

        # Kerasではこの関数に名前を付ける必要がある
        partial_gp_loss.__name__ = 'gradient_penalty'

        # 入力[本物の画像, ランダムノイズ], 出力[本物画像の予測値, 偽物画像の予測値, 補間画像の予測値]
        self.critic_model = Model(inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated])

        self.critic_model.compile(loss=[wasserstein, wasserstein, partial_gp_loss], optimizer=self.optimizer_c, loss_weights=[1, 1, self.grad_weight])


        """ 生成器を訓練するモデルのコンパイル"""
        set_trainable(self.critic, False)
        set_trainable(self.generator, True)
        self.gan = self.build_GAN()
        self.gan.compile(optimizer=self.optimizer_g, loss=wasserstein)

    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def train_discriminator(self, x_train):
        
        valid = np.ones((self.batch_size, 1), dtype=np.float32)
        fake = -np.ones((self.batch_size, 1), dtype=np.float32)
        dummy = np.zeros((self.batch_size, 1), dtype=np.float32) # gradient_penalty_lossのためのダミーラベル

        idx = np.random.randint(0, x_train.shape[0], self.batch_size)
        valid_imgs = x_train[idx]
        noise = np.random.normal(0, 1, (self.batch_size, self.z_dim))
        # 入力[本物の画像, ランダムノイズ], 出力[本物画像のラベル, 偽物画像のラベル, 補間画像のダミーのラベル(実際には使われない)]
        d_loss = self.critic_model.train_on_batch([valid_imgs, noise], [valid, fake, dummy])

        return d_loss


    def train_generator(self):
        
        valid = np.ones((self.batch_size, 1))
        noise = np.random.normal(0, 1, (self.batch_size, z_dim))
        self.gan.train_on_batch(noise, valid)

    def train_on_batch(self, x_train, epochs):
        
        for epoch in range(epochs):
            print('\repoch : %d' %(epoch), end='')
            self.train_discriminator(x_train, self.batch_size)
            self.train_generator(self.batch_size)


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

gan = GAN(img_shape, z_dim, batch_size)
gan.compile()
gan.train_on_batch(dataset, batch_size, epochs)
