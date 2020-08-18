# 変数、メソッドについてはp115

import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

# 二層のニューラルネット
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size=10, weight_init_std=0.01):
    # input_size:入力層の数 hidden_size:隠れ層の数 output_size:出力層の数 weight_init_std:学習率
    # 今回は数字画像の認識のため入力と出力数は指定しておく

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x:入力データ t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t) # 損失関数の値（交差エントロピー誤差）
    
    # 正確さ
    def accuracy(self, x, t):
        y = predict(x)
        y = np.argmax(y, axis=1) # 各行の最大値のインデックスの配列を作る
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0]) # x.shape[0]はxの行の数
        return accuracy

    # 各重みとバイアスの勾配を求めparamに対応するようにgradsに格納
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    