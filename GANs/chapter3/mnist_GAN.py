import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python import keras as K
from tensorflow.python.keras.datasets import mnist




# 生成器
def build_generator(img_shape, z_dim):

    model = K.models.Sequential()
    # 全結合層
    model.add(K.layers.Dense(128, input_dim=z_dim))
    # Leaky ReLUによる活性化
    model.add(K.layers.advanced_activations.LeakyReLU(alpha=0.01))
    # tanh関数を使った出力層
    model.add(K.layers.Dense(img_rows * img_cols * channels, activation='tanh'))
    # 生成器の出力が画像サイズになるようにreshape
    model.add(K.layers.Reshape(img_shape))

    return model

# 識別器
def build_discriminator(img_shape):

    model = K.models.Sequential()
    # 入力画像を一列に並べる
    model.add(K.layers.Flatten(input_shape=img_shape))
    # 全結合層
    model.add(K.layers.Dense(128))
    # Leaky ReLUによる活性化
    model.add(K.layers.advanced_activations.LeakyReLU(alpha=0.01))
    # sigmoid関数を通して出力
    model.add(K.layers.Dense(1, activation='sigmoid'))

    return model

# 生成器と識別器からGANを構成
def build_gan(generator, discriminator):

    model = K.models.Sequential()
    # 生成器と識別器の統合
    model.add(generator)
    model.add(discriminator)

    return model

def train(iterations, batch_size, sample_interval):
    a = 0
    # MNISTデータセットのロード
    (X_train, _), (_ , _) = mnist.load_data()

    # [0, 255]の範囲のグレースケールが措置を[-1, 1]にスケーリング
    X_train = X_train / 127.5 - 1.0
    # (60000, 28, 28) => (60000, 28, 28, 1)
    X_train = np.expand_dims(X_train, axis=3)

    # 本物の画像のラベルは全て1とする
    real = np.ones((batch_size, 1))

    # 偽の画像のラベルは全て0とする
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):

        #----------------------
        #　識別器の訓練
        #----------------------

        # 本物の画像をランダムに取り出したバッチを作る(0以上画像の枚数未満)
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # 偽の画像のバッチを生成する
        # zは正規分布からbatch_size個サンプリング
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # 識別器の訓練
        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss, accuracy = 0.5 * np.add(d_loss_real, d_loss_fake)


        #----------------------
        #　生成器の訓練
        #----------------------

        # 偽の画像のバッチを生成する
        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        # 生成器の訓練
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:

            # 訓練終了後に図示するために、損失と精度をセーブしておく
            losses.append((d_loss, g_loss))
            accuracies.append(100.0 * accuracy)
            iteration_checkpoints.append(iteration + 1)

            # 訓練の進捗を出力する
            print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' %(iteration+1, d_loss, 100.0*accuracy, g_loss))

            # 生成された画像のサンプルを出力
            sample_images(generator)


# sample_intervalおきに呼ばれ、訓練中に作成された画像を確認することができる
def sample_images(generator, image_grid_rows=4, image_grid_columns=4):

    # ランダムノイズのサンプリング
    z = np.random.normal(0, 1, (image_grid_rows * image_grid_columns, z_dim))

    # ランダムノイズを使って画像を生成[-1, 1]の範囲
    gen_imgs = generator.predict(z)

    # 画像の画素値を[0, 1]の範囲にスケール
    gen_imgs = 0.5 * gen_imgs + 0.5

    # 画像をグリッドに並べる
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharey=True, sharex=True)
    count = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            # 並べた画像を出力
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    plt.show()


img_rows = 28
img_cols = 28
channels = 1
# 生成器の入力として使われるノイズベクトルの次元
z_dim = 100
# ハイパーパラメータ
iterations = 30000
batch_size = 128
sample_interval = 1000

losses = []
accuracies = []
iteration_checkpoints = []

# 入力画像の次元
img_shape = (img_rows, img_cols, channels)

# 識別器の構築とコンパイル
discriminator = build_discriminator(img_shape)
discriminator.compile(optimizer=K.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 生成器の構築
generator = build_generator(img_shape, z_dim)

# 生成器の構築中は識別器のパラメータを固定
discriminator.trainable = False

# 生成器の訓練のため、識別機は固定し、GANモデルの構築とコンパイルを行う
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=K.optimizers.Adam())

# 設定した反復回数だけGANの訓練を行う
train(iterations, batch_size, sample_interval)
