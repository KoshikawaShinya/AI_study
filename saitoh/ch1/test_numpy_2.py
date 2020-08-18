import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("lena.png", 0) # C:\Users\81808\desktop\python\saitoh に移動しておく

plt.imshow(img)

plt.show()