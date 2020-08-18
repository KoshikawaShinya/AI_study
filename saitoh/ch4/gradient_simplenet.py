import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

# 一層のネットワーク
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 標準偏差1で初期化

    # 信号と重みの内積で予測
    def predict(self, x):
        return np.dot(x, self.W)

    # 損失関数の値を求める
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z) # zをsoftmax関数で正規化
        loss = cross_entropy_error(y, t) # 交差エントロピー誤差
        return loss