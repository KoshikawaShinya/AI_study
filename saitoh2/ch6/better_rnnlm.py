import sys
sys.path.append('..')
from common.time_layers import *
from common.np import *
from common.base_model import BaseModel
import numpy as np


"""
LSTMレイヤの多層化 (ここでは2層)
Dropoutを使用 (深さ方向にのみ適用)
重み共有 (EmbeddingレイヤとAffineレイヤで重み共有)
"""

class BetterRnnlm(BaseModel):
    
    def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_rate=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        
        # 重みの初期化
        # embeddingレイヤ以降の重みはXavierの初期値を使用
        embed_W = (np.random.randn(V, D) / 100).astype('f')

        lstm_Wx1 = (np.random.randn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')

        lstm_Wx2 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (np.random.randn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')

        # affineの重みは作らない
        affine_b = np.zeros(V).astype('f')

        # レイヤの生成
        # 3つの改善
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_rate),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_rate),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_rate),
            TimeAffine(embed_W.T, affine_b) # 重み共有!
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        # 全ての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
    # softmaxレイヤの直前までを処理
    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    # 引数に時系列データと教師ラベル
    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()
    