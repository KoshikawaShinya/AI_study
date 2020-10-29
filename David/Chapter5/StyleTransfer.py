from keras.applications import vgg19
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
import numpy as np

base_img_path = 'imgs/transfer/Content.jpg'
style_reference_img_path = 'imgs/transfer/Style.jpg'

content_weight = 1.0
style_weight = 100.0
total_variation_weight = 20.0
style_loss = 0.0

width, height = load_img(base_img_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# ベース画像とスタイル画像を保持する2つのKeras変数と、生成された統合画像を格納するプレースホルダの定義
base_img = K.variable(preprocess_image(base_img_path))
style_reference_img = K.variable(preprocess_image(style_reference_img_path))
combination_img = K.placeholder((1, img_nrows, img_ncols, 3))

# VGG19モデルへの入力テンソルは、3つの画像を連結したもの
input_tensor = K.concatenate([base_img, style_reference_img, combination_img], axis=0)

# imagenetを学習したvgg19のインスタンス化。include_top=False : 画像分類の結果を出力する全結合層を読み込まない
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

# 各層の出力をdict型で保存
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


""" コンテンツ損失 """
# コンテンツ損失に使うのは5ブロック目の2番目の畳み込み層
layer_features = outputs_dict['block5_conv2']

# ベース画像と統合画像の特徴を抜き出す
base_img_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

# コンテンツ損失はベース画像と統合画像の二乗和誤差
def content_loss(content, gen):
    return K.sum(K.square(gen - content))

content_loss = content_weight * content_loss(base_img_features, combination_features)


""" スタイル損失 """
# グラム行列
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# スタイル画像を統合画像のグラム画像で二乗和誤差を用い、類似性を見る
def cal_style_loss(style, combination):
    s = gram_matrix(style)
    c = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(s - c)) / (4.0 * (channels ** 2) * (size ** 2))

# スタイル損失には5つの層の出力を使用
feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# 各層のスタイル損失を計算し、足す
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = cal_style_loss(style_reference_features, combination_features)
    style_loss += (style_weight / len(feature_layers)) * sl


""" 全変動損失 """
# 1ピクセルずらし、元の画像都の二乗誤差を計算
def total_variation_loss(x):
    # 元の画像と1ピクセル下にずらした同じ画像との間の二乗誤差
    a = K.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, 1:, :img_ncols-1, :])
    # 元の画像と1ピクセル右にずらした同じ画像との間の二乗誤差
    b = K.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, :img_nrows-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

tv_loss = total_variation_weight * total_variation_loss(combination_img)

""" 全体の損失 """
loss = content_loss + style_loss + tv_loss

outputs = [loss]
f_outputs = K.function([combination_img], outputs)
def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

# evaluator : 入力画像に関して、全体的な損失と損失の勾配を計算するメソッドを持つインスタンス
evaluator = Evaluator()

iterations = 1000
x = preprocess_image(base_img_path)

for i in range(iterations):
    print(i)
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)