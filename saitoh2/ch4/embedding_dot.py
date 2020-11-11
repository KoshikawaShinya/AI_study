import sys
sys.path.append('..')
from common.layers import Embedding
import numpy as np

class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        # 出力側の重みからターゲットの分散表現を得る
        target_W = self.embed.forward(idx)
        # 中間層とターゲットの分散表現の要素ごとの積
        # axis=1となっているのはバッチ処理を想定しており、target_W, hのどちらも二次元配列となっているため
        # 詳しくはp150
        out = np.sum(target_W * h, axis=1)

        # 計算した値の一時保持
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
