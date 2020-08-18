import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

x = np.arange(0, 7, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
print(x)
print(y1)

plt.plot(x, y1, label="sin") # 線の描写と名前付け 
plt.plot(x, y2, linestyle="--", label="cos") # 破線の描写と名前
plt.xlabel("x") # x軸の名前
plt.ylabel("y") # y軸の名前
plt.title("sin & cos") # タイトル
plt.legend() # 線の名前と線の種類を左下
plt.show()