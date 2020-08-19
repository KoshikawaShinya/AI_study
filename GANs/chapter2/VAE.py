from tensorflow.python import keras as K
from tensorflow.python.keras.datasets import mnist
import tensorflow as tf
import numpy as np

batch_size = 100

# MNIST画像の高さｘ幅（784 = 28 x 28）
original_dim = 784
latent_dim = 2
intermediate_dim = 256

# エポック数
epochs = 50
epsilon_std = 1.0

# (args: tuple) : 型アノテート(引数の説明)
def sampling(args: tuple):
    # tupleから変数値を得る
    z_mean, z_log_var = args
    epsilon = K.backend.random_normal(shape=(K.backend.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.backend.exp(z_log_var / 2) * epsilon

# 第二の引数グループは入力の指定
# エンコーダへの入力
x = K.layers.Input(shape=(original_dim), name='input')
# 中間層
h = K.layers.Dense(intermediate_dim, activation='relu', name='encoding')(x)
# 潜在空間の平均(mean)を定義
z_mean = K.layers.Dense(latent_dim, name='mean')(h)
# 潜在空間でのlog分散(log variance)を定義
z_log_var = K.layers.Dense(latent_dim, name='log-variance')(h)

z = K.layers.Lambda(sampling, output_shape=(latent_dim))([z_mean, z_log_var])
# KerasのModelとしてエンコーダを定義
encoder = K.models.Model(x, [z_mean, z_log_var, z], name='encoder')


# デコーダへの入力
input_decoder = K.layers.Input(shape=(latent_dim,), name='decoder_input')
# 潜在空間から途中の次元数にする
decoder_h = K.layers.Dense(intermediate_dim, activation='relu', name='decoder_h')(input_decoder)
# 元の大きさの次元にデコードする
x_decoded = K.layers.Dense(original_dim, activation='sigmoid', name='flat_decoded')(decoder_h)
# デコーダをKerasのモデルとして定義
decoder = K.models.Model(input_decoder, x_decoded, name='decoder')
decoder.summary()

# 結果を得る。encoderの3つ目の要素はサンプリングされたz
output_combined = decoder(encoder(x)[2])
# 入力と、全体の出力を結び付ける
vae = K.models.Model(x, output_combined)
# モデル全体がどうなっているかを表示
vae.summary()

# 損失関数の定義
def vae_loss(x: tf.Tensor, x_decoded_mean: tf.Tensor, z_log_var=z_log_var, z_mean=z_mean, original_dim=original_dim):
    # バイナリ交差エントロピー
    xent_loss = original_dim * K.metrics.binary_crossentropy(x, x_decoded_mean)
    # KLダイバージェンス(二つの分布の距離を計算、分布が離れているほど大きな値をとる)
    kl_loss = -0.5 * K.backend.sum(1 + z_log_var - K.backend.square(z_mean) - K.backend.exp(z_log_var), axis=-1)
    # 二つの損失を加算して平均をとることで全体の損失とする
    vae_loss = K.backend.mean(xent_loss + kl_loss)
    return vae_loss

# 最後にモデルをコンパイルする
# rmsprop : 勾配の振れ幅が大きい時に学習率を小さくして値を更新する
vae.compile(optimizer='Adam', loss=vae_loss)
vae.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 0~1の間の値に正規化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# データを一列に並べ替え(28 x 28 => 784)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_train.shape[1:])))

vae.fit(x_train, x_train, shuffle=True, epochs=epochs, batch_size=batch_size)
vae.save('saved_model/mnist_vae.h5' )