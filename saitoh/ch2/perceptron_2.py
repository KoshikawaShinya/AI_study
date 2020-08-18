import numpy as np

# AND回路
def AND(x1, x2):
    x = np.array([x1, x2]) # 入力
    w = np.array([0.5, 0.5]) # 重み
    b = -0.7 # バイアス

    tmp = np.sum(x * w) + b 

    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# NAND回路
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    tmp = np.sum(x * w) + b

    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# OR回路
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2

    tmp = np.sum(x * w) + b

    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# EX-OR回路
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


print(AND(0, 1))
print(NAND(1, 1))
print(OR(0, 1))
print(XOR(1, 0))
        