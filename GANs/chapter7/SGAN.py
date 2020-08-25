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

# 識別器のコア部分
def build_discriminator_net(img_shape):

    model = Sequential()

    # 28x28x1 を 14x14x32 のテンソルに変換する畳み込み層
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))

    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 14x14x32 を 7x7x64 のテンソルに変換する畳み込み層
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))

    # バッチ正規化
    model.add(BatchNormalization())

    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 7x7x64 を 3x3x128 のテンソルに変換する畳み込み層
    model.add(Conv2D(128, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))

    # バッチ正規化
    model.add(BatchNormalization())

    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # ドロップアウト
    # バッチ正規化後に追加することで相互作用で良い結果をもたらす
    model.add(Dropout(0.5))

    # テンソルを一列に並べる
    model.add(Flatten())

    # num_classesニューロンへの全結合層
    model.add(Dense(num_classes))

    return model

# SGAN教師あり部分
def build_discriminator_supervised(discriminator_net):

    model = Sequential()

    model.add(discriminator_net)

    # softmax活性化関数。本物のクラス中のどれに該当するかの推定確率を出力
    model.add(Activation('softmax'))

    return model

# SGAN教師なし部分
def build_discriminator_unsupervised(discriminator_net):

    model = Sequential()

    model.add(discriminator_net)

    def predict(x):
        # 本物のクラスにわたる確率分布を、本物か偽物かの二値の確率に変換する
        prediction = 1.0 - (1.0 / (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))

        return prediction

    # 本物か偽物かを出力する、すでに定義済みのニューロン
    model.add(Lambda(predict))

    return model

# SGANモデルの構築
def build_gan(generator, discriminator):

    model = Sequential()

    # 生成器と識別器のモデルの統合
    model.add(generator)
    model.add(discriminator)

    return model

# 訓練
def train(iterations, batch_size, sample_interval):

    # 本物の画像につけられたラベル：全て１
    real = np.ones((batch_size, 1))
    # 偽物の画像につけられたラベル：全て０
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        #-------------
        # 識別器の訓練
        #-------------

        # ラベル付きのサンプルを得る
        imgs, labels = dataset.batch_labeled(batch_size)
        # ワンホットエンコードされたラベル
        labels = to_categorical(labels, num_classes=num_classes)

        # ラベルなしのサンプルを得る
        imgs_unlabeled = dataset.batch_unlabeled(batch_size)

        # 偽画像のバッチを作る
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # ラベル付きの本物サンプルによる訓練
        d_loss_supervised, accuracy = discriminator_supervised.train_on_batch(imgs, labels)

        # ラベルなしの本物サンプルによる訓練
        d_loss_real = discriminator_unsupervised.train_on_batch(imgs_unlabeled, real)
        # 偽のサンプルによる訓練
        d_loss_fake = discriminator_unsupervised.train_on_batch(gen_imgs, fake)

        d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)

        #-------------
        # 生成器の訓練
        #-------------

        # 偽画像のバッチを生成
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # 生成器を訓練する
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # 識別器の教師あり分類損失を、訓練後にプロットするために保存しておく
            supervised_losses.append(d_loss_supervised)
            iteration_checkpoints.append(iteration + 1)

            # 訓練の進捗を出力
            print('%d [D loss supervised: %.4f, acc.: %.2f%%][D loss unsupervised: %.4f][G loss: %f]'\
                %(iteration+1, d_loss_supervised, 100*accuracy, d_loss_unsupervised, g_loss))



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

supervised_losses = []
iteration_checkpoints = []

# ハイパーパラメータ
iterations = 8000
batch_size = 32
sample_interval = 800

dataset = Dataset(num_labeled)

# 識別器ネットワークのコア部分：
# この層は教師ありと教師なしの訓練中に共有される
discriminator_net = build_discriminator_net(img_shape)

# 教師あり学習のための識別器を作ってコンパイルする
discriminator_supervised = build_discriminator_supervised(discriminator_net)
discriminator_supervised.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())

# 教師なし学習のための識別器を作ってコンパイルする
discriminator_unsupervised = build_discriminator_unsupervised(discriminator_net)
discriminator_unsupervised.compile(loss='binary_crossrntropy', optimizer=Adam())

# 生成器の作成
generator = build_generator(z_dim)

# 生成器の訓練中は識別器のパラメータは定数としておく
discriminator_unsupervised.trainable = False

# 生成器の訓練を行うため、
# 識別器のパラメータを固定したGANモデルを構築し、コンパイルする
gan = build_gan(generator, discriminator_unsupervised)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# SGANを、指定した回数だけ反復訓練
train(iterations, batch_size, sample_interval)


# 精度のチェック
x, y = dataset.test_set()
y = to_categorical(y, num_classes=num_classes)
# テストデータの分類制度を計算
_, accuracy = discriminator_supervised.evaluate(x, y)
print('Test Accuracy: %.2f%%' %(100 * accuracy))
