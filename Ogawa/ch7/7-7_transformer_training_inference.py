from torch import nn, optim
import torch

from utils.dataloader import get_IMDb_Dataloaders_and_TEXT
from utils.transformer import TransformerClassification

# 読み込み
train_dl, val_dl, test_dl, TEXT = get_IMDb_Dataloaders_and_TEXT(max_length=256, batch_size=64)

# 辞書オブジェクトにまとめる
dataloaders_dict = {'train' : train_dl, 'val' : val_dl}

# モデル構築
net = TransformerClassification(text_embedding_vectors = TEXT.vocab.vectors, d_model=300, max_seq_len=256, output_dim=2)

# ネットワークの初期化を定義
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # Linear層の初期化
        nn.init.kaiming_normal
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# 訓練モードに設定
net.train()

# TransformerBlockモジュールを初期化実行
net.net3_1.apply(weights_init)
net.net3_2.apply(weights_init)

print('ネットワーク設定完了')

# 損失関数の設定
criterion = nn.CrossEntropyLoss()
# nn.LogSoftmax()を計算してからnn.NLLLoss(negative log likehood loss)を計算

# 最適化手法の設定
learning_rate = 2e-5
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかの確認
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用デバイス : ', device)
    print('-------start-------')
    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化
    torch.backends.cudnn.benchmark = True

    # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() # モデルを訓練モードに
            else:
                net.eval()  # モデルを検証モードに
            
            epoch_loss = 0.0    # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # データローダーからミニバッチを取り出すループ
            for batch in dataloaders_dict[phase]:
                # batchはTextとLabelの辞書オブジェクト

                # GPUが使えるならGPUにデータを送る
                inputs = batch.Text[0].to(device)   # 文章
                labels = batch.Label.to(device)     # ラベル

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝播(forward)計算
                with torch.set_grad_enabled(phase == 'train'):

                    # mask作成
                    input_pad = 1   # 単語IDにおいて、'<pad>' : 1のため
                    input_mask = (inputs != input_pad)

                    # Transformerに入力
                    outputs, _, _ = net(inputs, input_mask)
                    loss = criterion(outputs, labels)   # 損失を計算

                    _, preds = torch.max(outputs, 1)    # ラベルを予想

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # loss 合計を更新
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)
            
            # epochごとのlossと正解率
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} | Loss : {:.4f} Acc : {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc))

    return net


