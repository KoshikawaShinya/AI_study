import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Concatenate, Dense, Dropout, Flatten, Input, Lambda, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical


# MNISTデータセットを半教師あり学習に対応させるためのclass
class Dataset:
    def __init__(self, num_labeled):
        
        # 訓練に用いるラベル付き訓練データの数
        # num_labeledによってMNIST訓練データの一部をラベル付きデータとして訓練する
        self.num_labeled = num_labeled

        # MNISTデータセットのロード
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        def process_imgs(x):
            # [0, 255]のグレースケール画素値を[-1, 1]に変換
            x = (x.astype(np.float32) - 127.5) / 127.5
            # 画像の次元を 横幅 x 縦幅 x チャンネル数 に拡張する
            x = np.expand_dims(x, axis=-1)
            return x
        
        def process_labels(y):
            return y.reshape(-1, 1)

        # 訓練データ
        self.x_train = process_imgs(self.x_train)
        self.y_train = process_labels(self.y_train)

        # 検証データ
        self.x_test = process_imgs(self.x_test)
        self.y_test = process_labels(self.y_test)

    def batch_labeled(self, batch_size):
        # ラベル付き画像と、ラベル自身をランダムに取り出してバッチを作る
        idx = np.random.randint(0, self.num_labeled, batch_size)
        imgs = self.x_train[idx]
        labels = self.y_train[idx]
        return imgs, labels

    def batch_unlabeled(self, batch_size):
        # ラベルなし画像からランダムなバッチを作る
        idx = np.random.randint(self.num_labeled, self.x_train.shape[0], batch_size)
        imgs = self.x_train[idx]
        return imgs

    def training_set(self):
        x_train = self.x_train[range(self.num_labeled)]
        y_train = self.y_train[range(self.num_labeled)]
        return x_train, y_train

    def test_set(self):
        return self.x_test, self.y_test

# 生成器
def build_generator(z_dim):

    model = Sequential()

    # 入力を全結合層を通じて7x7x256のテンソルに変形する
    model.add(Dense(256 * 7 * 7, input_dim=z_dim))
    model.add(Reshape((7, 7, 256)))

    # 7x7x256 を 14x14x128 びテンソルに変換する転置畳み込み層
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))

    # バッチ正規化
    model.add(BatchNormalization())

    # Leaky ReLu による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 14x14x128 を 14x14x64のテンソルに変換する転置畳み込み層
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))

    # バッチ正規化
    model.add(BatchNormalization())

    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 14x14x64 を 28x28x1 のテンソルに変換する転置畳み込み層
    model.add((Conv2DTranspose(1, kernel_size=3, strides=2, padding='same')))

    # tanh活性化関数を用いた出力層
    model.add(Activation('tanh'))

    return model

# 識別器
def build_discriminator_net(img_shape):

    model = Sequential()

    # 28x28x1 を 14x14x32 のテンソルに変換する畳み込み層
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))

    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 14x14x32 を 7x7x64 のテンソルに変換する畳み込み層
    model.add(Conv2D(64, ketnel_size=3, strides=2, input_shape=img_shape, padding='same'))

    # バッチ正規化
    model.add(BatchNormalization())

    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 7x7x64 を 3x3x128 のテンソルに変換する畳み込み層
    model.add(Conv2D(128, ketnel_size=3, strides=2, input_shape=img_shape, padding='same'))

    # バッチ正規化
    model.add(BatchNormalization())

    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # ドロップアウト
    model.add(Dropout(0.5))

    # テンソルを一列に並べる
    model.add(Flatten())

    # num_classesニューロンへの全結合層
    model.add(Dense(num_classes))

    return model


img_rows = 28
img_cols = 28
channels = 1

# 入力画像の解像度
img_shape = (img_rows, img_cols, channels)

# 生成器の入力として使われるノイズベクトルのサイズ
z_dim = 100

# データセット内のクラスの数
num_classes = 10
# ラベル付きデータの数(それ以外はラベル無しとして使われる)
num_labeled = 100

dataset = Dataset(num_labeled)