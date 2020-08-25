import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

with tf.Graph().as_default():
    # TFHubからプログレッシブGANをimport
    module = hub.Module('https://tfhub.dev/google/progan-128/1')
    # ランタイムにサンプルされる潜在ベクトルの次元
    latent_dim = 512

    # 違う顔を生成する場合はseedを変更する
    latent_vector = tf.random_normal([1, latent_dim], seed=200)

    # モジュールを使って、latent spaceから画像を生成。実装の細部はオンラインにある
    interpolated_images = module(latent_vector)
    
    # Tensorflow session を走らせ、(1, 128, 128, 3)次元の画像を得る
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        image_out = session.run(interpolated_images)

plt.imshow(image_out.reshape(128,128,3))
plt.show()
