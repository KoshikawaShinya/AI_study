from keras.applications import vgg19
from keras import backend as K
from keras.preprocessing.image import load_img

base_img_path = ''
style_reference_img_path = ''

content_weight = 0.01

# ベース画像とスタイル画像を保持する2つのKeras変数と、生成された統合画像を格納するプレースホルダの定義
base_img = K.variable(load_img(base_img_path))
style_reference_img = K.variable(load_img(style_reference_img_path))
combination_img = K.placeholder((1, img_n_rows, img_n_cols, 3))

# VGG19モデルへの入力テンソルは、3つの画像を連結したもの
input_tensor = K.concatenate([base_img, style_reference_img, combination_img], axis=0)

# imagenetを学習したvgg19のインスタンス化。include_top=False : 画像分類の結果を出力する全結合層を読み込まない
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

# 各層の出力をdict型で保存
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# コンテンツ損失に使うのは5ブロック目の2番目の畳み込み層
layer_features = outputs_dict['block5_conv2']

# ベース画像と統合画像の特徴を抜き出す
base_img_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

# コンテンツ損失はベース画像と統合画像の二乗和誤差
def content_loss(content, gen):
    return K.sum(K.square(gen - content))

content_loss = content_weight * content_loss(base_img_features, combination_features)