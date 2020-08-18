import numpy as np

x = np.array([[1.0, 2.0], [2.0, 4.0], [4.0, 16.0]])
y = np.array([[2.0, 4.0], [4.0, 8.0]])
print(x[0][1])
a = x.flatten()
print(a)
print(a[a>15])
print(x.shape)