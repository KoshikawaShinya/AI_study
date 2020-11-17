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

class TimeLSTM:

    def __init__(self, Wx, Wh, b, stateful=false):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        # バッチ数, 時系列データ数, 単語ベクトルの次元
        N, T, D = xs.shape
        # 隠れ状態ベクトルの次元数
        H = Wh.shape[0]

        self.layers = []
        # LSTMレイヤの各時刻の出力を格納するための
        hs = np.empty((N, T, H), dtype='f')

        # statefulがFalseとなっている場合か。隠れ状態と記憶セルに何も入っていない場合初期化
        # statefulをFalseにすると初期化が起こり、前のTimeLSTMで最後に計算された隠れ状態と記憶セルを記憶しなくなる
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)
        
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            # dhは出力側と隣の層に分かれているため、足す
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        # 前の層への勾配
        self.dh = dh

        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None
