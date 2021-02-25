import glob
import os
import io
import string
import re
import random
import torchtext
from torchtext.vocab import Vectors

# 以下の記号をスペースに置き換える (カンマ、ピリオドを除く)
# punctuationとは日本語で句点という意味
print('区切り文字 : ', string.punctuation)


# 前処理

def preprocessing_text(text):
    # 改行コードを消去
    text = re.sub('<br />', '', text)

    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == '.') or (p == ','):
            continue
        else:
            text = text.replace(p, ' ')
    
    # ピリオドなどの前後にはスペースを入れておく
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    return text

# 分かち書き
def tokenizer_punctuation(text):
    return text.strip().split()

# 前処理と分かち書きをまとめた関数を定義
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    ret = tokenizer_punctuation(text)
    return ret


# 文章とラベルの両方に用意
max_length = 256
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True,
                            batch_first=True, fix_length=max_length, init_token='<cls>', eos_token='<eos>')
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
# 引数の意味
# init_token : 全部の文章で、文頭に入れておく単語
# eos_token  : 全部の文章で、文末に入れておく単語

# フォルダ「data」から各tsvファイルを読み込む
train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
    path='./data/', train='IMDb_train.tsv', test='IMDb_test.tsv', format='tsv',
    fields=[('Text', TEXT), ('Label', LABEL)]
)

train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))

english_fasttext_vectors = Vectors(name='data/wiki-news-300d-1M.vec')
TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=10)

# Dataloaderを作成
train_dl = torchtext.data.Iterator(train_ds, batch_size=24, train=True)
val_dl = torchtext.data.Iterator(val_ds, batch_size=24, train=False, sort=False)
test_dl = torchtext.data.Iterator(test_ds, batch_size=24, train=False, sort=False)

# 動作を確認
print(tokenizer_with_preprocessing('I like cats.'))
print('訓練及び検証のデータ数 : ', len(train_val_ds))
print('1つ目の訓練及び検証のデータ : ', vars(train_val_ds[0]))
# 単語ベクトルの中身を確認
print('1単語を表現する次元数 : ', english_fasttext_vectors.dim)
print('単語数 : ', len(english_fasttext_vectors.itos))
# ボキャブラリーのベクトルを確認
#print(TEXT.vocab.vectors.shape)
#print(TEXT.vocab.stoi)