import sys
sys.path.append('..')
from common.time_layers import TimeEmbedding, TimeLSTM
import numpy as np


class Encoder:

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        embed_W = (np.random.randn(V, D) / 100).astype('f')
        lstm_Wx = (np.random.randn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        # 今回はTimeLSTMは状態を保持しないためstatefulをFalseとする
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs

        # 最後のLSTMレイヤが出力したhだけが必要
        return hs[:, -1, :]
    
    def backward(self, dh):
        # デコーダー側からの勾配dhしか勾配は存在しない
        dhs = np.zeros_like(self.hs)
        # backwardを使用するにはdhsの形にする必要があるためdhをdhsの該当する箇所に設定する
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)

        return dout
