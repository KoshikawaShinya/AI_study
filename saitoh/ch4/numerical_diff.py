import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4 # 0.0001 このくらいなら丸目誤差が発生しない
    return (f(x+h) - f(x-h)) / (2 * h)

# 二次元の配列
def function_1(x):
    return 0.01*x**2 + 0.1*x

# 多次元の関数
def function_2(x):
    return np.sum(x**2)

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
