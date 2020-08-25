import tensorflow as tf
import keras as K
import numpy as np


def upscale_layer(layer, upscale_factor):
    '''
    テンソルが[group, height, width, channel]のときに、
    層（テンソル）をfactor（int）分だけアップスケールする
    '''

    height, width = layer.get_shape()[1:3]
    size = (upscale_factor * height, upscale_factor * width)
    upscaled_layer = tf.image.resize_nearest_neighbor(layer, size)
    return upscaled_layer

def smooyhly_maerge_last_layer(list_of_layers, alpha):
    '''
    alphaに基づいて、スムーズに層を混ぜ込むこの関数は、全ての層が
    すでにRGBになっていると仮定する
    これは生成器のための関数
    list_of_layers : サイズで順序付けされたテンソルを要素に持つ
    alpha          : [0, 1]の間の浮動小数値
    '''
    # ヒント！　もしKerasではなく純粋なtensorflowを使っている場合は、scopeを忘れないように
    last_fully_trained_layer = list_of_layers[-2]

    # もともと学習されている層を生成している
    last_layer_upscaled = upscale_layer(last_fully_trained_layer, 2) 

    # まだ学習されていない、新たに加えられる層
    larger_native_layer = list_of_layers[-1]

    # 層を混ぜるためのコードが実行できるかの確認
    assert larger_native_layer.get_shape() == last_layer_upscaled.get_shape()

    # この部分はブロードキャストを利用するとよい
    new_layer = (1 - alpha) * last_layer_upscaled + alpha * larger_native_layer

    return new_layer

def minibatch_std_layer(layer, group_size=4):
    '''
    ここでは、入力された層のミニバッチ標準偏差を計算する
    Kerasを使い、設定済みのtf-scopeの中で処理を行う
    layerのデータ型はfloat32だと仮定する。そうでない場合はバリデーション/キャストが必要
    ノート:Kerasではもっと効率の良い方法がある
    ここでは実装を明快にすることと、理解促進のためのメジャーな実装に近づけるために、
    今の方法にしています。これは練習問題だと思ってください。
    '''
    # ヒント！
    # もしKerasではなくて純粋なTensorflowを使っている場合は、scopeを忘れないようにしてください。
    # 1つのミニバッチグループは、group_sizeで割り切れるか、またはgroup_size以下でなければならない。
    group_size = K.backend.minimum(group_size, tf.shape(layer)[0])

    # デフォルト値の確保と後のアクセスを容易にするため、shape情報を取っておく
    # 画像本体の各次元の配列数はグラフ実行前にNoneとしてキャストされることが
    # 一般的なので、tf_shapeから入力を取得する
    shape = list(K.int_shape(input))
    shape[0] = tf.shape(input)[0]

    # ミニバッチレベルで処理を行うためのreshape()です
    # このコードではレイヤは[グループ(G),ミニバッチ(M),幅(W),高さ(H),チャンネル(C)]
    # で表現されることとしていますが、他の実装ではTheanoに沿った順序に
    # なっていることもあるので注意してください
    minibatch = K.backend.reshape(layer, (group_size, -1, shape[1], shape[2], shape[3]))

    # グループ[M,W,H,C]全体にわたって、元データから平均を引いて中心にそろえる
    minibatch -= tf.reduce_mean(minibatch, axis=0, keepdims=True)
    # グループ[M,W,H,C]の分散を求める
    minibatch = tf.reduce_mean(K.backend.square(minibatch), axis=0)
    # グループ[M,W,H,C]の標準偏差を求める
    minibatch = K.backend.square(minibatch + 1e8)
    # RGB画像の全てのピクセルの平均をとることで標準偏差の平均値が得られる
    minibatch = tf.reduce_mean(minibatch, axis=[1, 2, 4], keep_dims=True)
    # 得られたスカラ値で、画像の元の高さとチャンネル数のサイズを埋める
    minibatch = K.backend.tile(minibatch, [group_size, 1, shape[2], shape[3]])
    # 新しい特徴マップとして後ろに追加する
    return K.backend.concatenate([layer, minibatch], axis=1)


def equalize_learning_rate(shape, gain, fan_in=None):
    '''
    Heの初期化法から得られる定数分だけ、全層の重みを修正する
    これにより、異なった特徴のダイナミックレンジに応じて分散を調整できる
    shape: テンソル（レイヤ）のshape
        例えば[4, 4, 48, 3]となる
        この場合は[カーネルのサイズ,カーネルのサイズ,フィルタの数,特徴マップ]を意味する
        ただ、これは他の部分の実装によって少しずつ異なる
    
    gain:   通常はsqurt(2)
    fan_in: XavierまたはHeによる初期化ごとの、入力数の修正量
    '''
    # デフォルト値は、shapeのすべての次元数をかけたものから特徴マップの次元数を引いたもの
    # この数はニューロン一つあたりの入力数になります
    if fan_in is None: fan_in = np.prod(shape[:-1])
    # Heの初期値を使う (He et al, 2015)
    std = gain / K.sqrt(fan_in)
    # 修正量とは無関係に定数を作る
    wscale = K.constant(std, name='wscale', dtype=np.float32)
    # 重みを取得してから、ブロードキャストを使って更新量を適用する
    adjueted_weights = K.get_value('layer', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    return adjusted_weights


# ピクセル正規化
def pixelwise_feat_norm(inputs, **kwargs):
    '''
    ピクセルごとの特徴正規化[Krizhevsky et at. 2012]
    入力を正規化して返す
    inputs: Keras/TFのLayers
    '''
    normalization_constant = K.backend.sqrt(K.backend.mean(inputs**2, axis=-1, keepdims=True) + 1.0e-8)
    return inputs / normalization_constant


