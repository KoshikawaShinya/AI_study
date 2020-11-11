import numpy as np


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    # 重みからidx番目を抜き取ることで単語の分散表現を得る
    def forward(self, idx):
        W, =self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        # dWのすべての要素に0を代入
        dW[...] = 0
        
        # dW配列のself.idx番目にdoutをaddする
        # idxの要素が重複した場合に代入だと片方が上書きされてしまう。そのため「加算」
        np.add.at(dW, self.idx, dout)

        return None
