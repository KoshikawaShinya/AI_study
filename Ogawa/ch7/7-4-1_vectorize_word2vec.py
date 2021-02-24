from janome.tokenizer import Tokenizer
from gensim.models import KeyedVectors
from torchtext.vocab import Vectors
import MeCab
import torchtext
import torch.nn.functional as F
import re

j_t = Tokenizer()
m_t = MeCab.Tagger('-Owakati')

def tokenizer_mecab(text):
    text = m_t.parse(text)
    return text.strip().split()

def tokenizer_janome(text):
    return[tok for tok in j_t.tokenize(text, wakati=True)]

# 前処理として正規化する関数
def preprocessing_text(text):
    # 半角・全角の統一
    # 今回は無視

    # 英語の小文字化
    # 今回は無視
    # output = output.lower()

    # 改行、半角スペース、全角スペースを削除
    text = re.sub('\r', '', text)
    text = re.sub('\n', '', text)
    text = re.sub(' ', '', text)
    text = re.sub('　', '', text)

    # 数字文字の一律「0」化
    text = re.sub(r'[0-9 ０-９]', '0', text)    # 数字

    # 記号と数字の除去
    # 今回は無視

    # 特定文字を正規表現で置換
    # 今回は無視

    return text

# 前処理とJanomeの単語分割を合わせた関数を定義する
def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text) # 文章の正規化
    ret = tokenizer_janome(text)    # Janomeの単語分割

    return ret

max_length = 25
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True,
                            batch_first=True, fix_length=max_length)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(path='./data/', train='text_train.tsv', validation='text_val.tsv',
                                                                 test='text_test.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])

# entity_vector.model.binはそのままではtorchtextで扱えないため、gensimで読み込んで、torchtextで扱える形で保存しなおす。
#model = KeyedVectors.load_word2vec_format('./data/entity_vector/entity_vector.model.bin', binary=True)
#model.wv.save_word2vec_format('./data/japanese_word2vec_vectors.vec')

japanese_word2vec_vectors = Vectors(name='./data/japanese_word2vec_vectors.vec')

# 単語ベクトルの中身を確認
print('1単語を表現する次元数 : ', japanese_word2vec_vectors.dim)
print('単語数 : ', len(japanese_word2vec_vectors.itos))

# ベクトル化したバージョンのボキャブラリを作成
TEXT.build_vocab(train_ds, vectors=japanese_word2vec_vectors, min_freq=1)

# ボキャブラリーのベクトルを確認
print(TEXT.vocab.vectors.shape)     # 49個の単語が200次元のベクトルで表現されている
print(TEXT.vocab.vectors)

# ボキャブラリーの順番を確認
print(TEXT.vocab.stoi)

# 姫 - 女性 + 男性のベクトルがどれと似ているか確認
tensor_calc = TEXT.vocab.vectors[41] - TEXT.vocab.vectors[38] + TEXT.vocab.vectors[46]

# コサイン類似度を計算
print('女王', F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[39], dim=0))
print('王', F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[44], dim=0))
print('王子', F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[45], dim=0))
print('機械学習', F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[43], dim=0))