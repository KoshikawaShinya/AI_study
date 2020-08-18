#p92~

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

# 二乗和誤差
def mean_squared_error(y,t):
    e = 0.5 * np.sum((y - t)**2)
    return e

#交差エントロピー誤差 バッチに対応＆tがone-hot表現でないものに対応
def cross_entropy_error(y, t):
    if y.ndim == 1: # もしバッチで送られてこなかった場合
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]　# 送られてきたバッチのサイズ
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 0 ~ 6000 の間の数をランダムに10個選んだ配列を作る
x_batch = x_train[batch_mask] # batch_maskでランダムに選ばれたインデックスを指定して10個のミニバッチとする
t_batch = t_train[batch_mask]


print(x_batch.shape)
print(t_batch.shape)