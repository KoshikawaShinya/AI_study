from __future__ import print_function, division
import scipy
from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys, os
import numpy as np
from glob import glob


class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob('./datasets/%s/%sA/*' % (self.dataset_name, data_type))
        path_B = glob('./datasets/%s/%sB/*' % (self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)


class CycleGAN:
    def __init__(self):
        # 入力のshape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # データローダの設定
        self.dataset_name = 'apple2orange'
        # 前処理済みのデータをインポートするためにDataLoaderオブジェクトを用いる
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_rows, self.img_cols))

        # D(PatchGAN)の出力shapeを計算する
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Gの最初の層のフィルタ数
        self.gf = 32
        # Dの最初の層のフィルタ数
        self.df = 64

        # Loss weights
        # サイクル一貫性損失の重み(どれだけ厳密にサイクル一貫性損失を考慮するか。大きくすると元の画像と再編成した画像が可能な限り似たものとなる)
        self.lambda_cycle = 10.0
        # 同一性損失の重み(同一性損失に影響を与える。小さくすると不要な変化が起こりやすくなる)
        self.lambda_id = 0.9 * self.lambda_cycle

        optimizer = Adam(0.0002, 0.5)

        # 識別器の構築とコンパイル
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        #-------------------------------------
        # ここからは生成器の計算グラフを構築する
        #-------------------------------------

        # 生成器の作成
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # 両ドメインから来た画像を入力する
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # 画像を他のドメインに翻訳する
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # 画像を元のドメインに再翻訳する
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # 画像の恒等写像
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # 複合モデルに対して、生成器のみを訓練する
        self.d_A.trainable = False
        self.d_B.trainable = False

        # 翻訳した画像の妥当性を識別器が決定する
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # 複合モデルによって生成器を訓練して、識別器をだませるようにする
        self.combined = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])
        # mse : 平均二乗誤差、 mae : 平均絶対誤差
        self.combined.compile(loss=['mse','mse','mae','mae','mae','mae'],
                              loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id],
                              optimizer=optimizer)


    def build_generator(self):
        """U-Netの生成"""

        def conv2d(layer_input, filters, f_size=4, normalization=True):
            """識別器"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)

            return d

        @staticmethod
        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """アップサンプリング中に使われる層##"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])

            return u


        # 画像入力
        # d0(128 x 128 x 3)
        d0 = Input(shape=self.img_shape)

        # ダウンサンプリング
        # d0(128 x 128 x 3) => d1(64 x 64 x 32)
        d1 = self.conv2d(d0, self.gf)
        # d1(64 x 64 x 32) => d2(32 x 32 x 64)
        d2 = self.conv2d(d1, self.gf*2)
        # d2(32 x 32 x 64) => d3(16 x 16 x 128)
        d3 = self.conv2d(d2, self.gf*4)
        # d3(16 x 16 x 128) => d4(8 x 8 x 256)
        d4 = self.conv2d(d3, self.gf*8)

        # アップサンプリング
        # d4(8 x 8 x 256) => u1(16 x 16 x 256)
        u1 = self.deconv2d(d4, d3, self.gf*4)
        # u1(16 x 16 x 256) => u2(32 x 32 x 128)
        u2 = self.deconv2d(u1, d2, self.gf*2)
        # u2(32 x 32 x 128) => u3(64 x 64 x 64)
        u3 = self.deconv2d(u2, d1, self.gf)

        # u3(64 x 64 x 64) => u4(128 x 128 x 64)
        u4 = UpSampling2D(size=2)(u3)
        # u4(128 x 128 x 64) => (128 x 128 x 3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)


    # 識別器の構築
    def build_discriminator(self):
        # 入力画像(128 x 128 x 3)
        img = Input(shape=self.img_shape)

        # img(128 x 128 x 3) => d1(64 x 64 x 64)
        d1 = self.conv2d(img, self.df, normalization=False)
        # d1(64 x 64 x 64) => d2(32 x 32 x 128)
        d2 = self.conv2d(d1, self.df*2)
        # d2(32 x 32 x 128) => d3(16 x 16 x 256)
        d3 = self.conv2d(d2, self.df*4)
        # d3(16 x 16 x 256) => d4(8 x 8 x 512)
        d4 = self.conv2d(d3, self.df*8)

        # d4(8 x 8 x 512) => (8 x 8 x 1)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)


# CycleGANの訓練
class CycleGAN(CycleGAN):
    def train(self, epochs, batch_size=1, sample_interval=50):
        # 敵対性損失の正解ラベル
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                #-------------
                # 識別器の訓練
                #-------------

                # ここから識別器を訓練する。この行は画像を逆のドメインに翻訳する
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # 識別器を訓練する(元画像=real / 翻訳されたもの=fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # 識別器の誤差全体
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                #-------------
                # 生成器の訓練
                #-------------

                # 生成器の訓練
                # 損失(識別器によるAの妥当性, 識別器によるBの妥当性, 再翻訳されたA画像, 再翻訳されたB画像, Aの恒等画像, Bの恒等画像)
                #                       敵対性損失                           サイクル一貫性損失                同一性損失
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])

                # 保存インターバルになったら生成された画像サンプルを保存
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)