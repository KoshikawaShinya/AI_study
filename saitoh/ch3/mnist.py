# p72 ~ 
import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image # PIL(Python Image Libraly)モジュール

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # Numpy配列をPIL用のデータオブジェクトに変える
    pil_img.show()

# load_mnist関数は (訓練画像, 訓練ラベル), (テスト画像, テストラベル) を戻り値として返してくる
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False) # 引数についてはp74

img = x_train[1]
label = t_train[1]
print(label)

print(img.shape) # flattenをTrueにしているため画像データが一列に格納されている  
img = img.reshape(28,28) # 一列になっている画像を28*28の画像に戻す
print(img.shape)

img_show(img)