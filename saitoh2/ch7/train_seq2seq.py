import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq


# データセットの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()

# ハイパーパラメータの設定
vocab_sie = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# モデル/オプティマイザ/トレーナの生成
model = PeekySeq2seq(vocab_sie, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 学習
acc_list = []
for epoch in range(max_epoch):
    # エポックごとにテストデータを使い正解率を求める
    # そのためfitの引数のmax_epochを1とする
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        # vervoseはテストデータの最初の10個の間Trueとなる
        verbose = i < 10
        # eval_seq2seq : 問題をモデルに与え、文字列生成を行わせ、それがあっているかどうかを判定する。あっていれば1、間違っていれば0を返す
        # verbose=Trueの時、結果がターミナルに表示される
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

