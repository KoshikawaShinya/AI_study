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
