#p89

import numpy as np

# 二乗和誤差
def mean_squared_error(y, t): # 引数はNumpy配列
    e = 0.5 * np.sum((y - t)**2)
    return e

#交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    e = -1 * np.sum(t * np.log(y + delta)) # np.log(0)になったとき負の無限大となってしまうため微笑量deltaを足して計算する
    return e

t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) # 正解は2
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]) # 2の確率が0.6 
print(mean_squared_error(y, t))
print(cross_entropy_error(y, t))

y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.5, 0.0]) # 2の確率が0.1
print(mean_squared_error(y, t))
print(cross_entropy_error(y, t))