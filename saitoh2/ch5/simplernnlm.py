import sys
sys.path.append('..')
import numpy as np
from common.time_layers import *

class SimpleRnnlm:

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
    
        # 重みの初期化
        # RNNレイヤとaffineレイヤにXavierの初期値を用いる
        embed_W = (np.random.randn(V, D) / 100).astype('float32')
        rnn_Wx = (np.random.randn(D, H) / np.sqrt(D)).astype('float32')
        rnn_Wh = (np.random.randn(H, H) / np.sqrt(H)).astype('float32')
        rnn_b = np.zeros(H).astype('float32')
        affine_W = (np.random.randn(H, V) / np.sqrt(H)).astype('float32')
        affine_b = np.zeros(V).astype('float32')

        # レイヤの生成
        # Truncated BPTT を使用するためTimeRNNレイヤのstatefulをTrueにする
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        # 全ての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)

        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout
    
    def reset_state(self):
        self.rnn_layer.reset_state()