import numpy as np
import matplotlib.pylab as plt

# Numpy配列に対応したステップ関数
def step_function(x):
    y = x > 0 # 配列の値を0を基準にbool型に変換
    return y.astype(np.int) # bool型の値をint型に変換 [faise = 0, true = 1]

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1) #y軸の範囲
plt.show()