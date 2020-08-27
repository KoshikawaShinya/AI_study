import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Concatenate, Dense, Embedding, Flatten, Input, Multiply, Reshape, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam


# GANの生成器
def build_generator(z_dim):

    model = Sequential()

    # 入力を、全結合層を通じて 7x7x256のテンソルに変形する
    model.add(Dense(7*7*256, input_dim=z_dim))
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


def build_generator_conv2d(z_dim):

    model = Sequential()

    # 入力を、全結合層を通じて 7x7x256のテンソルに変形する
    model.add(Dense(7*7*256, input_dim=z_dim))
    model.add(Reshape((7, 7, 256)))

    # 7x7x256 を 14x14x128 びテンソルに変換する転置畳み込み層
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    # バッチ正規化
    model.add(BatchNormalization())
    # Leaky ReLu による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 14x14x128 を 14x14x64のテンソルに変換する転置畳み込み層
    model.add(Conv2D(64, kernel_size=3, padding='same'))
    # バッチ正規化
    model.add(BatchNormalization())
    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 14x14x64 を 28x28x1 のテンソルに変換する転置畳み込み層
    model.add(UpSampling2D())
    model.add((Conv2D(1, kernel_size=3, padding='same')))

    # tanh活性化関数を用いた出力層
    model.add(Activation('tanh'))

    return model


# CGANの生成器
def build_cgan_generator(z_dim):

    # ランダムノイズベクトルz
    z = Input(shape=(z_dim, ))

    # 条件ラベル : Gが生成しなければならない番号を指定する0-9の整数
    label = Input(shape=(1, ), dtype='int32')

    # ラベル埋め込み：
    #----------------
    # ラベルをz_dim次元の密ベクトルに変換する：
    # (batch_size, 1, z_dim次元の3次元ベクトルになる)
    label_embedding = Embedding(num_classes, z_dim, input_length=1)(label)

    # 埋め込みを行った3Dテンソルを(batch_size, z_dim)次元を持つ2DテンソルへとFlattenする
    label_embedding = Flatten()(label_embedding)

    # ベクトルzとラベルが埋め込まれたベクトルの、要素ごとの掛け算を行う
    joined_representation = Multiply()([z, label_embedding])

    generator = build_generator(z_dim)

    # 与えられたラベルを持つ画像を生成
    conditioned_img = generator(joined_representation)

    return Model([z, label], conditioned_img)

# GANの識別器
def build_discriminator(img_shape):

    model = Sequential()

    # 28x28x2 を 14x14x64のテンソルに変換する畳み込み層
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(img_shape[0], img_shape[1], img_shape[2] + 1), padding='same'))
    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 14x14x64 を 7x7x64のテンソルに変換する畳み込み層
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    # バッチ正規化
    model.add(BatchNormalization())
    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 7x7x64 を 3x3x128 のテンソルに変換する畳み込み層
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    # バッチ正規化
    model.add(BatchNormalization())
    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # シグモイド関数を適用した出力層
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

# CGANの識別器
def build_cgan_discriminator(img_shape):

    # 入力画像
    img = Input(shape=img_shape)

    # 入力画像に対するラベル
    label = Input(shape=(1, ), dtype='int32')

    # ラベル埋め込み：
    #----------------
    # ラベルをz_dim次元の密ベクトルに変換する：
    # (batch_size, 1, 28x28x1)の形の3Dテンソルを生成する
    label_embedding = Embedding(num_classes, np.prod(img_shape), input_length=1)(label)

    # embeddingな3Dテンソルを、(batch_size, 28x28x1)の形の2Dテンソルに展開する
    label_embedding = Flatten()(label_embedding)

    # ラベル埋め込みを、入力画像と同じ形に変形する
    label_embedding = Reshape(img_shape)(label_embedding)

    # 画像にラベル埋め込みを結合する
    concatenated = Concatenate(axis=-1)([img, label_embedding])

    discriminator = build_discriminator(img_shape)

    # 画像ーラベルの組を分類する
    classification = discriminator(concatenated)

    return Model([img, label], classification)

# CGANモデル
def build_cgan(generator, discriminator):

    # ランダムなノイズベクトルz
    z = Input(shape=(z_dim, ))

    # 画像ラベル
    label = Input(shape=(1, ))

    # そのラベルに対して生成された画像
    img = generator([z, label])

    classification = discriminator([img, label])

    # 生成器 -> 識別器とつながる統合モデル
    # G([z, label]) = x*
    # D([x*, label]) = 分類結果
    model = Model([z, label], classification)

    return model

def train(iterations, batch_size, sample_interval):

    # MNISTデータセットを読み込む
    (x_train, y_train), (_, _) = mnist.load_data()

    # グレースケールのが措置を、[0, 255]から[-1, 1]にスケーリング
    x_train = x_train.astype(np.float32) / 127.5 - 1.0
    x_train = np.expand_dims(x_train, axis=-1)

    # 本物の画像のラベルを全部1にする
    real = np.ones((batch_size, 1))
    # 偽物の画像のラベルを全部0にする
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        #-------------
        # 識別器の訓練
        #-------------

        # 本物の画像とラベルの組からなるバッチをランダムに作る
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs, labels = x_train[idx], y_train[idx]

        # 偽の画像からなるバッチを生成する
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict([z, labels])

        # 識別器を訓練する
        d_loss_real = discriminator.train_on_batch([imgs, labels], real)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        #-------------
        # 生成器の訓練
        #-------------

        # ノイズベクトルからなるバッチを生成する
        z = np.random.normal(0, 1, (batch_size, z_dim))
        # ランダムなラベルを持つバッチを生成する
        labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

        # 生成器を訓練する
        g_loss = cgan.train_on_batch([z, labels], real)

        print('\rNo, %d' %(iteration+1), end='')

        if (iteration + 1) % sample_interval == 0:

            # 訓練過程を出力
            print(' [D loss: %f, acc.: %.2f%%] [G loss: %f]' %( d_loss[0], 100*d_loss[1], g_loss))

            # 後でプロットするために損失と精度を保存する
            losses.append((d_loss[0], g_loss))
            accuracies.append(100 * d_loss[1])

            # 生成した画像を出力する
            sample_images()


def sample_images(image_grid_rows=2, image_grid_columns=5):

    # ノイズベクトルを生成する
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # 0-9の画像のラベルを得る
    labels = np.arange(0, 10).reshape(-1, 1)
    
    # ノイズベクトルから画像を生成する
    gen_imgs = generator.predict([z, labels])

    # 出力の画素値を[0, 1]の範囲にスケーリングする
    gen_imgs = 0.5 * gen_imgs + 0.5

    # 画像からなるグリッドを生成する
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(10, 4), sharey=True, sharex=True)
    count = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # 並べた画像を出力
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title('Digit: %d' % labels[count])
            count += 1
    



img_rows = 28
img_cols = 28
channels = 1
# 入力画像の次元
img_shape = (img_rows, img_cols, channels)
# 生成器の入力として用いるノイズベクトルのサイズ
z_dim = 100
# データセットに含まれるクラス数
num_classes = 10

accuracies = []
losses = []

# ハイパーパラメータ
iterations = 12000
batch_size = 32
sample_interval = 100

# 識別器の構築とコンパイル
discriminator = build_cgan_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 生成器の構築
generator = build_cgan_generator(z_dim)
# 生成器の訓練においては識別器のパラメータは定数とする
discriminator.trainable = False

# 生成器を訓練するために、識別器の(パラメータは)個令して、CGANモデルの構築とコンパイルを行う
cgan = build_cgan(generator, discriminator)
cgan.compile(loss='binary_crossentropy', optimizer=Adam())

# CGANを決められた回数、反復訓練する
train(iterations, batch_size, sample_interval)