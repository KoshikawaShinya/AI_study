import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.util import im2col

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T # フィルターの展開 -1は要素数のつじつまが合うようにまとめてくれる機能
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # transposeは多次元配列の順番をインデックスで指定することで入れ替える

        return out

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 展開（１）
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        print(col.shape)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        print(col.shape)

        # 最大値（２）
        out = np.max(col, axis=1)
        # 整形（３）
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out