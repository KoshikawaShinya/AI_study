#p67 ~ 70
import numpy as np

def softmax_1(a): # aの値が大きいと正しく計算されなくなる (オーバーフロー)
    exp_a = np.exp(a) # 分子
    sum_exp_a = np.sum(exp_a) # 分母
    y = exp_a / sum_exp_a
    return y

def softmax_2(a): # softmax関数をコンピュータ用に改善したもの
    c = np.max(a)
    exp_a = np.exp(a - c) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a1 = np.array([0.3, 2.9, 4.0])
a2 = np.array([1010, 1000, 990])

y1 = softmax_1(a1)
y2 = softmax_1(a2)
print(y1)
print(y2)

y1 = softmax_2(a1)
y2 = softmax_2(a2)
print(y1)
print(y2)
