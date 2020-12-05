import sys
sys.path.append('..')
from common.layers import Softmax
import numpy as np

# 学習するパラメータを持たない
class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
    
    def forward(self, hs, a):
        N, T, H = hs.shape
        
        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.cache
        N, T, H  = hs.shape

        # dc.shape is (N, H)
        # sumの逆伝播
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        
        dar = dt * hs
        dhs = dt * ar

        # repeatの逆伝播
        da = np.sum(dar, axis=2)

        return dhs, da


class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        N, T, H = hs.shape

        # h = (N, H)
        
        # (N, T, H)
        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        # (N, T, H)
        t = hs * hr
        # (N, T)
        s = np.sum(t, axis=2)
        # (N, T)
        a = self.softmax.forward(s)

        self.cache = (hs, hr)
        return a

    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape

        # (N, T)
        ds = self.softmax.backward(da)
        # (N, T, H)
        dt = dt.reshape(N, T, 1).repeat(H, axis=2)
        # (N, T, H)
        dhs = dt * hr
        dhr = dt * hs
        # (N, T)
        dh = np.sum(dhr, axis=1)

        return dhs, dh

# Weight Sum レイヤと Attention Weight レイヤによる順伝播、逆伝播を行う
class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out

    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh


class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        # 各時間のAttention
        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh
        
        return dhs_enc, dhs_dec