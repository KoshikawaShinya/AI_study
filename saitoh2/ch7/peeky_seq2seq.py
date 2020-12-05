import sys
sys.path.append('..')
from common.time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss
from common.base_model import BaseModel
from ch7.seq2seq import Seq2seq, Encoder
import numpy as np


# 学習時にのみsoftmaxを使用し、生成時にはsoftmaxを使用しないため、Decoderとしてはsoftmaxを実装しない。
class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        # 重みの初期化
        # 大体Xavierの初期値
        # lstmとaffineの重みの形状を隠れ状態を連結した入力に対応するように変更
        embed_W = (np.random.randn(V, D) / 100).astype('f')
        lstm_Wx = (np.random.randn(H+D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (np.random.randn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (np.random.randn(H+H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # レイヤの生成
        self.embed = TimeEmbedding(embed_W)
        # encoderの出力hをdecoderのlstmレイヤに設定するためstatefulはTrue
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []

        # パラメータと勾配をメンバ変数にまとめる
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

        self.cache = None

    # 学習時のみ使用
    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape

        # lstmの初期の隠れ状態にhを設定
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        # hを時系列分だけ複製
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        # Embeddingレイヤの出力outとhを複製したhsをnp.concatenate()で連結
        out = np.concatenate((hs, out), axis=2)

        out = self.lstm.forward(out)
        # lstmレイヤの出力outとhを複製したhsをnp.concatenate()で連結
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)
        self.cache = H
        return score

    # 上方向にあるSoftmaxWithLossレイヤから勾配dscoreを受け取る
    def backward(self, dscore):
        H = self.cache

        dout = self.affine.backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled




class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size

        # エンコーダ、デコーダの生成
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        # パラメータと勾配をまとめる
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads



