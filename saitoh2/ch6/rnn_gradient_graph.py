import numpy as np
import matplotlib.pyplot as plt

# ミニバッチサイズ
N = 2
# 隠れ状態ベクトルの次元数
H = 3
# 時系列データの長さ
T = 20

# 上流からの勾配
dh = np.ones((N, H))
# 再現性のために乱数のシードを固定
np.random.seed(3)

# Whの初期値
#Wh = np.random.randn(H, H) # 勾配爆発
Wh = np.random.randn(H, H) * 0.5 # 勾配消失

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    # L2ノルムによって、dhの大きさを求めている
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

# プロット
plt.plot(norm_list)
plt.ylabel('norm')
plt.xlabel('time step')
plt.show()
