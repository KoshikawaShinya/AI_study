from janome.tokenizer import Tokenizer
import torchtext
import re

j_t = Tokenizer()

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

# 動作確認
text = '昨日は とても暑く、気温が　36度もあった。'
print(tokenizer_with_preprocessing(text))

# tsvやcsvデータを読み込んだ時に、読み込んだ内容に対して行う処理の定義
# 文章とラベルの両方に用意

max_length = 25
TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True, lower=True, include_lengths=True,
                            batch_first=True, fix_length=max_length)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

# フォルダ「data」から各tsvファイルを読み込み、Datasetにする
# 1行がTEXTとLABELで区切られていることをfieldsで指示する
train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(path='./data/', train='text_train.tsv', validation='text_val.tsv',
                                                                 test='text_test.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])

# 動作確認
print(len(train_ds))
print(vars(train_ds[0]))
print(vars(train_ds[1]))

# ボキャブラリーを作成
# 訓練データtrainの単語からmin_freq以上の頻度の単語を使用してボキャブラリー(単語集)を構築
TEXT.build_vocab(train_ds, min_freq=1)

# 訓練データ内の単語路頻度を出力(頻度min_freqより大きいものが出力される)
print(TEXT.vocab.freqs)
print(TEXT.vocab.stoi)

# DataLoaderを作成 (torchtextの文脈では単純にiteratorと呼ばれている)
train_dl = torchtext.data.Iterator(train_ds, batch_size=2, train=True)
val_dl = torchtext.data.Iterator(val_ds, batch_size=2, train=False, sort=False)
test_dl = torchtext.data.Iterator(test_ds, batch_size=2, train=False, sort=False)

# 動作確認 検証データのデータセットで確認
batch = next(iter(val_dl))
print(batch.Text)
print(batch.Label)
