import numpy as np

dW1 = np.random.randn(3, 3) * 10
dW2 = np.random.randn(3, 3) * 10
grads = [dW1, dW2]
max_norm = 5.0

def clip_grads(grads, max_norm):
    total_norm = 0
    print("クリッピング前 : ")
    print(dW1)
    
    # 全てのパラメータに対する勾配のL2ノルムを計算
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    # max_normよりL2ノルムのほうが大きくなる場合、rateを各勾配に掛けることで勾配爆発を防止する
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

    print("クリッピング後 : ")
    print(grads[0])

clip_grads(grads, max_norm)