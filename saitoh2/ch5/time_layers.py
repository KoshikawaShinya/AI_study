import numpy as np

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params

        # 入力xと前の層からの入力h_prevで各重みと行列積を行いバイアスを足し、tanh関数で変換
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        # tanhとバイアスの逆伝播
        dtanh = dh_next * (1 - h_next ** 2)
        db = np.sum(dtanh, axis=0)
        # 行列積の逆伝播
        dWh = np.dot(h_prev.T, dtanh)
        dh_prev = np.dot(dtanh, Wh.T)
        dWx = np.dot(x.T, dtanh)
        dx = np.dot(dtanh, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return dx, dh_prev

# RNNをT個連結し、T個の時系列データを処理する
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        # self.h : 次のTimeRNNに送る隠れ状態  self.dh : 前時刻に送る隠れ状態の勾配。今回は使用しない
        self.h, self.dh = None, None
        # stateful : 隠れ状態を引き継ぐかどうか
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        # N : バッチ数  T : 時系列データの長さ  D : 入力ベクトルの次元数  H : 隠れ状態の次元数
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        # self.statefulがFalseの時、またはself.hに何もない時
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        # T回、レイヤの生成と順伝播を繰り返す
        # 各RNNレイヤでは同じ重みを使用
        for t in range(T):
            # *self.paramsは [Wx, Wh, b] を Wx, Wh, b にアンパックする
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]

        # reversed : 逆順に取りだす
        for t in reversed(range(T)):
            layer = self.layers[t]
            # 出力側で分岐するため、合算された勾配が入力となる
            dx, dh = layer.backward(dhs[:, t, :] + dh) # 合算した勾配
            dxs[:, t, :] = dx

            # 各レイヤで同じ重みを使用しているため、重みの勾配を加算
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        # 最終的な結果をself.gradsに三点ドットを使って上書き
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs
