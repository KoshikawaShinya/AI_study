import glob
import os
import io
import string
import re

# 訓練データのtsvファイルを作成

f = open('./data/IMDb_train.tsv', 'w', encoding="utf-8")

path = './data/aclImdb/train/pos/'

for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding='utf-8') as ff:
        text = ff.readline()

        # タブがあれば消しておく
        text = text.replace('\t', ' ')

        text = text+'\t'+'1'+'\t'+'\n'
        f.write(text)

path = './data/aclImdb/train/neg/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding='utf-8') as ff:
        text = ff.readline()

        # タブがあれば消しておく
        text = text.replace('\t', ' ')

        text = text+'\t'+'0'+'\t'+'\n'
        f.write(text)
f.close()

# テストデータのtsvファイルを作成

f = open('./data/IMDb_test.tsv', 'w', encoding='utf-8')

path = './data/aclImdb/test/pos/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding='utf-8') as ff:
        text = ff.readline()

        # タブがあれば消しておく
        text = text.replace('\t', ' ')

        text = text+'\t'+'1'+'\t'+'\n'
        f.write(text)

path = './data/aclImdb/test/neg/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding='utf-8') as ff:
        text = ff.readline()

        # タブがあれば消しておく
        text = text.replace('\t', ' ')

        text = text+'\t'+'0'+'\t'+'\n'
        f.write(text)
f.close()
