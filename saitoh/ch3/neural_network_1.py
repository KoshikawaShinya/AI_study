""" 三層のニューラルネット
    入力層が28×28の画像のため784個
    一つ目の隠れ層が50個
    二つ目の隠れ層が100個
    出力層が0~9の10個のニューロンを持つ
"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


# 入力テストデータと出力テストデータを返す
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 学習済みの重みとバイアスのパラメータをディクショナリ型の変数として返す
def init_network():
    with open("sample_weight.pkl", 'rb') as f: # sample_weight.pklをrb(バイナリファイル読み込み)しfとする
        network = pickle.load(f) # fを非直列化しpythonオブジェクトをファイルから復元
    return network

# 入力された画像から分類をし確率を返す
def predict(network, x):
    W1, W2, W3 = network["W1"],network["W2"],network["W3"]
    b1, b2, b3 = network["b1"],network["b2"],network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0 # 正確さカウント用変数
for i in range(len(x)): # xに入っている画像データの個数ぶん回す
    y = predict(network, x[i]) # 画像を判別し確率の配列を出す
    p = np.argmax(y) # もっとも確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy : " + str(float(accuracy_cnt / len(x))))