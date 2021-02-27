from torch import nn
import torch
import math

from utils.dataloader import get_IMDb_DataLoaders_and_TEXT

class Embedder(nn.Module):
    '''idで示されている単語をベクトルに変換'''

    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors, freeze=True)
        # freeze=Trueによりバックプロパゲーションで更新されず変化しなくなります

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec

class PositionalEncoder(nn.Module):
    '''入力された単語の位置を示すベクトル情報を付加する'''
    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()

        self.d_model = d_model  # 単語ベクトルの次元数

        # 単語の順番(pos)と埋め込みベクトルの次元の位置(i)によって一意に定まる値の表をpeとして作成
        pe = torch.zeros(max_seq_len, d_model)

        # GPUが使える場合はGPUへ送る
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))

        # 表peの先頭に。ミニバッチ次元となる次元を足す
        self.pe = pe

        # 勾配を計算しないようにする
        self.pe.requires_grad = False

    def forward(self, x):

        # 入力xとPositional Encodingを足し算する
        # xがpeよりも小さいので、大きくする
        ret = math.sqrt(self.d_model)*x + self.pe
        return ret


# 動作確認
train_dl, val_dl, test_dl, TEXT = get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=24)

# ミニバッチの用意
batch = next(iter(train_dl))

# モデル構築
net1 = Embedder(TEXT.vocab.vectors)
net2 = PositionalEncoder(d_model=300, max_seq_len=256)

# 入出力
x = batch.Text[0]
x1 = net1(x)
x2 = net2(x1)

print('入力のテンソルサイズ : ', x1.shape)
print('出力のテンソルサイズ : ', x2.shape)

