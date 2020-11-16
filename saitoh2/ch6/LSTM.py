import sys
sys.path.append('..')
import numpy as np
from common.functions import sigmoid


class LSTM:
    def __init__(self, Wx, Wh, b):
        # Wx、Whには各ゲートに使う4つの重みがまとめて格納されている。
        # 行列の要素数としては、Wx[D, 4H] Wh[H, 4H] となっている。
        # D : 単語ベクトルの次元数  H : 隠れ状態のベクトルの次元数
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        # バッチ数, 隠れ状態のベクトルの次元
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        # まとめて格納されている各ゲートに使う4つのパラメータをslice
        forget_gate = A[:, :H]
        g = A[:, H:2*H]
        input_gate = A[:, 2*H:3*H]
        output_gate = A[:, 3*H:]

        # ゲートをsigmoid関数に通し割合を決め、新しく記憶セルに追加する情報であるgはtanh関数に通す
        forget_gate = sigmoid(forget_gate)
        g = np.tanh(g)
        input_gate = sigmoid(input_gate)
        output_gate = sigmoid(output_gate)

        # cとhについてゲートとの要素積を行い、データを通す割合を決める
        c_next = c_prev * forget_gate + g * input_gate
        h_next = np.tanh(c_next) * output_gate

        self.cache = (x, h_prev, c_prev, input_gate, forget_gate, g, output_gate, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, input_gate, forget_gate, g, output_gate, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * output_gate) * (1 - tanh_c_next ** 2)

        dc_prev = ds * forget_gate

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * input_gate

        di *= input_gate * (1 - input_gate)
        df *= forget_gate * (1 - forget_gate)
        do *= output_gate * (1 - output_gate)
        dg *= (1 - g ** 2)

        # sliceした各値を一つにまとめる
        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev
