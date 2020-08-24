import tensorflow as tf
import keras as K


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