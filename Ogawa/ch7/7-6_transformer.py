from torch import nn

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

# 動作確認
train_dl, val_dl, test_dl, TEXT = get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=24)

# ミニバッチの用意
batch = next(iter(train_dl))

# モデル構築
net1 = Embedder(TEXT.vocab.vectors)
# 入出力
x = batch.Text[0]
x1 = net1(x)

print('入力のテンソルサイズ : ', x.shape)
print('出力のテンソルサイズ : ', x1.shape)

