import numpy as np

# 確率的勾配降下法
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[keys] -= self.lr * grads[key]

# 運動量を使った方法
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9): # mometum変数は摩擦や抵抗のようなもの
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items(): # .itemsはdict型のkeyとvalueに対してforループ処理
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
             params[key] += self.v[key]

# 学習係数を調整しながら学習を行う方法
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)