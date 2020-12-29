import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms

# 学習済みのVGG-16モデルをロード

# VGG-16モデルのインスタンスを生成
use_pretrained = True # 学習済みパラメータを使用
net = models.vgg16(pretrained=use_pretrained)
net.eval() # 推論モードに設定

# モデルのネットワーク構成を出力
print(net)

class BaseTransform():
    """
    画像のサイズをリサイズし、色を標準化する。

    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ
    mean : (R, G, B)
        各職チャネルの平均値
    std : (R, G, B)
        各色チャネルの標準偏差
    """

    def __init__(self, resize, mean, std):
        self.base_trainsform = transforms.Compose([
            transforms.Resize(resize), # 短い辺の長さがresizeの大きさになる
            transforms.CenterCrop(resize), # 画像の中央をresize x resizeで切り取り
            transforms.ToTensor(), # Torchテンソルに変換
            transforms.Normalize(mean, std) # 色情報の標準化
        ])

    def __call__(self, img):
        return self.base_trainsform(img)

# 画像毎処理の動作確認

# 1. 画像読み込み
image_file_path = './data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path)

# 2. 元の画像の表示
plt.imshow(img)
plt.show()

# 3. 画像の前処理と処理済み画像の表示
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img) # torch.size([3, 244, 244])

# (色、高さ、幅)を(高さ、幅、色)に変換し、0-1に値を制限して表示
img_transformed = img_transformed.numpy().transpose((1, 2, 0))
img_transformed = np.clip(img_transformed, 0, 1)
plt.imshow(img_transformed)
plt.imsave("./data/transformed.jpg", img_transformed)
plt.show()
