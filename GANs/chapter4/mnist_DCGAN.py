import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten, Reshape
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam



# 生成器
def build_generator(z_dim):

    model = Sequential()

    # 全結合層によって、7x7x256のテンソルに変換
    model.add(Dense(256 * 7 * 7, input_dim=z_dim))
    model.add(Reshape((7, 7, 256)))

    # 転置畳み込み層により、 7 x 7 x 256 を 14 x 14 x 128 のテンソルに変換
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    # バッチ正規化
    model.add(BatchNormalization())
    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 転置畳み込み層により、 14 x 14 x 128 を 14 x 14 x 64 のテンソルに変換
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    # バッチ正規化
    model.add(BatchNormalization())
    # Leaky ReLu による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 転置畳み込み層により、 14 x 14 x 64 を 28 x 28 x 1 のテンソルに変換
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
    # tanh関数を適用して出力
    model.add(Activation('tanh'))

    return model


# 識別器
def build_discriminator(img_shape):

    model = Sequential()

    # 28 x 28 x 1 を 14 x 14 x 32 のテンソルにする畳み込み層
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    # 14 x 14 x 32 を 7 x 7 x 64 のテンソルにする畳み込み層
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    # バッチ正規化
    model.add(BatchNormalization())
    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 7 x 7 x 64 を 3 x 3 x 128 のテンソルにする畳み込み層
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    # バッチ正規化
    model.add(BatchNormalization())
    # Leaky ReLU による活性化
    model.add(LeakyReLU(alpha=0.01))

    # 出力にシグモイド関数を適用
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


# GAN
def build_gan(generator, discriminator):

    model = Sequential()

    # 生成器と識別器のモデルを組み合わせる
    model.add(generator)
    model.add(discriminator)

    return model


# 学習ループ
def train(iterations, batch_size, sample_interval):

    losses = []
    accuracies = []
    iteration_checkpoints = []

    save_file = 0

    # mniseデータセットのロード
    (x_train, _), (_, _) = mnist.load_data()

    # [0, 255]の範囲の画素値を、[-1, 1]に変換
    x_train = x_train / 127.5 - 1.0
    # (60000, 28, 28) => (60000, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=3)

    # 本物の画像のラベルは全て1にする
    real = np.ones((batch_size, 1))

    # 偽物の画像のラベルは全て0にする
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        #-------------
        # 識別器の学習
        #-------------

        # 本物の画像集合からランダムにバッチを生成する
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        # 偽物の画像からなるバッチを生成する
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # 識別器の学習
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real , d_loss_fake)

        #-------------
        # 生成器の学習
        #-------------

        # ノイズベクトルを生成
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)
        
        # 生成器の学習
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # あとで可視化するために損失と精度を保存しておく
            losses.append([d_loss, g_loss])
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # 学習結果の出力
            print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' %(iteration+1, d_loss, 100.0*accuracy, g_loss))

            # 生成したサンプル画像を保存する
            save_file += sample_interval
            sample_images(generator, save_file, iteration+1)



def sample_images(generator, save_file, iteration, image_grid_rows=4, image_grid_columns=4):

    # ノイズベクトルを生成する
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # ノイズベクトルから画像を生成する
    gen_imgs = generator.predict(z)

    # 出力の画素値を[0, 1]の範囲にスケーリングする
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig = plt.figure()

    # 画像からなるグリッドを生成する
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharey=True, sharex=True)
    count = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # 並べた画像を出力
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    
    save_file += sample_interval
    fig.savefig('predict_imgs/{}itertions_trained.png'.format(iteration))


img_rows = 28
img_cols = 28
channels = 1

# ハイパーパラメータの設定
iterations = 100
batch_size = 64
sample_interval = 10


# 入力画像の解像度
img_shape = (img_rows, img_cols, channels)

# 生成器への入力として使われるノイズベクトルのサイズ
z_dim = 100

# 識別器の構築とコンパイル
discriminator = build_discriminator(img_shape)
discriminator.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 生成器の構築
generator = build_generator(z_dim)

# 生成器の訓練時のために、識別器のパラメータを固定
discriminator.trainable = False

# 識別器は固定したまま、生成器を訓練するGANモデルの生成とコンパイル
gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(), loss='binary_crossentropy')

train(iterations, batch_size, sample_interval)