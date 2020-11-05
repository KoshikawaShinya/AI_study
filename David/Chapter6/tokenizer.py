from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import re
import numpy as np

# 訓練させる系列データの長さ
seq_length = 20

file_name = 'aesop.txt'
new_file_name = 'aesop_fixed.txt'

with open(file_name, encoding='utf-8-sig') as f:
    text = f.read()

# 物語の始まりの改行の置き換え用
start_story = '| ' * seq_length

""" 前処理 """
text = start_story + text
# 大文字を小文字に変換
text = text.lower()
# 改行の置き換え
text = text.replace('\n\n\n\n\n', start_story)
text = text.replace('\n', ' ')
# 両端の空白文字を消す
text = re.sub('  +', '. ', text).strip()
# ..を.に置き換え
text = text.replace('..', '.')

# 正規表現による置換
text = re.sub('([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])', r' \1', text)
text = re.sub('\s{2,}', ' ', text)

with open(new_file_name, encoding='utf-8-sig', mode='w') as f:
    f.write(text)

""" トークン化 """
tokenizer = Tokenizer(char_level=False, filters='')
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
token_list = tokenizer.texts_to_sequences([text])[0]

# データセットの生成
def generate_sequences(token_list, step):

    X = []
    Y = []

    for i in range(0, len(token_list) - seq_length, step):

        X.append(token_list[i : i + seq_length])
        Y.append(token_list[i + seq_length])

    # one-hotベクトル化
    Y = to_categorical(Y, num_classes = total_words)

    num_seq = len(X)
    print('Number of sequences:', num_seq, "\n")

    return X, Y, num_seq

step = 1
seq_length = 20
X, y, num_seq = generate_sequences(token_list, step)

X = np.array(X)
Y = np.array(Y)