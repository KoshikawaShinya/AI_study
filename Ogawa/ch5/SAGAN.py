import torch
from torch import nn, optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import time


def make_datapath_list():
    """学習、検証の画像データとアノテーションデータへのファイルパスリストを作成"""

    train_img_list = list() # 画像ファイルパスを格納

    for img_idx in range(200):
        img_path = "./data/img_78/img_7" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)

        img_path = "./data/img_78/img_8" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)
    
    return train_img_list

class ImageTrandform():
    """画像の前処理クラス"""

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, img):
        return self.data_transform(img)
    
class GAN_Img_Dataset(data.Dataset):
    """画像のDatasetクラス。PyTorchのDatasetクラスを継承"""

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        """画像の枚数を返す"""
        return len(self.file_list)

    def __getitem__(self, index):
        """前処理をした画像のTensor形式のデータを取得"""

        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅]白黒

        # 画像の前処理
        img_transformed = self.transform(img)

        return img_transformed

class Self_Attention(nn.Module):
    """ Self-AttentionのLayer """

    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        
        # 1x1の畳み込み層によるpointwise convolutionを用意
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Attention Map作成時の規格化のソフトマックス
        self.softmax = nn.Softmax(dim=-2)

        # 元の入力xとSelf-Attention Mapであるoを足し算するときの係数
        # output = x + gamma * o
        # 最初はgamma = 0 で学習させていく
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        # 入力変数
        X = x

        # 畳み込みをしてから、サイズを変形する
        # (B, C', W, H) => (B, C', N)
        proj_query = self.query_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3]) # サイズ : (B, C', N)
        proj_query = proj_query.permute(0, 2, 1)    # 転置操作
        proj_key = self.key_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3]) # サイズ : (B, C', N)

        # 掛け算
        S = torch.bmm(proj_query, proj_key) # bmmはバッチごとの行列積

        # 規格化
        attention_map_T = self.softmax(S)   # 行i方向の和を1にするソフトマックス関数
        attention_map = attention_map_T.permute(0, 2, 1)    # 転置をとる

        # Self-Attention Mapを計算する
        proj_value = self.value_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3]) # サイズ : (B, C, N)
        o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))   # Attention Map は転置して掛け算

        # Self-Attention Map である o のテンソルサイズを X にそろえて、出力にする
        o = o.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
        out = x + self.gamma * o

        return out, attention_map
    
class Generator(nn.Module):

    def __init__(self, z_dim=20, image_size=64):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            # Spectral Normalizationを追加
            nn.utils.spectral_norm(nn.ConvTranspose2d(z_dim, image_size*8, kernel_size=4, stride=1)),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            # Spectral Normalizationを追加
            nn.utils.spectral_norm(nn.ConvTranspose2d(image_size*8, image_size*4, kernel_size=4, stride=2, paddig=1)),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            # Spectral Normalizationを追加
            nn.utils.spectral_norm(nn.ConvTranspose2d(image_size*4, image_size*2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True)
        )

        # Self-Attention層を追加
        self.self_attention1 = Self_Attention(in_dim=image_size*2)

        self.layer4 = nn.Sequential(
            # Spectral Normalizationを追加
            nn.utils.spectral_norm(nn.ConvTranspose2d(image_size*2, image_size, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True)
        )

        # Self-Attention層を追加
        self.self_attention2 = Self_Attention(in_dim=image_size)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(image_size, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        # 白黒画像なので出力チャネルは1

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map1 = self.self_attention1(out)
        out = self.layer4(out)
        out, attention_map2 = self.self_attention2(out)
        out = self.last(out)

        return out, attention_map1, attention_map2

class Discriminator(nn.Module):

    def __init__(self, z_dim=20, image_size=64):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, image_size, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # 白黒画像のため、入力チャネルは1つだけ

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(image_size, image_size*2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(image_size*2, image_size*4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Self-Attention層を追加
        self.self_attention1 = Self_Attention(in_dim=image_size*4)

        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(image_size*4, image_size*8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Self-Attention層を追加
        self.self_attention2 = Self_Attention(in_dim=image_size*8)

        self.last = nn.Conv2d(image_size*8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map1 = self.self_attention1(out)
        out = self.layer4(out)
        out, attention_map2 = self.self_attention2(out)
        out = self.last(out)
        
        return out, attention_map1, attention_map2

def train_model(G, D, dataloader, num_epochs):

    # GPUが使えるか確認
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用デバイス : ', device)

    # 最適化手法の設定
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    # 誤差関数を定義 -> hinge version of adversarial loss に変更
    # criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # パラメータをハードコーディング
    z_dim = 20
    mini_batch_size = 64

    # ネットワークをGPUへ
    G.to(device)
    D.to(device)

    # ネットワークをGPUへ
    G.train()
    D.train()

    # ネットワークがある程度固定であれば高速化
    torch.backends.cudnn.benchmark = True

    # 画像の枚数
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # イテレーションカウンタをセット
    iteration = 1
    logs = []

    # epochのループ
    for epoch in range(num_epochs):

        # 開始時刻を保存
        t_epoch_start = time.time()
        epoch_g_loss = 0.0  # epochの損失和
        epoch_d_loss = 0.0  # epochの損失和

        print('----------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('----------------')
        print(' (train) ')

        # データローダーからminibatchずつ取り出すループ
        for images in dataloader:

            # ----------------------
            # 1. Discriminatorの学習
            # ----------------------
            # ミニバッチがサイズ1だと、バッチノーマライゼーションでエラーになるので、避ける
            if images.size()[0] == 1:
                continue

            # GPUが使えるならGPUにデータを送る
            images = images.to(device)

            # 正解ラベルと偽ラベルを作成
            # epochの最後のイテレーションはミニバッチの数が少なくなる
            mini_batch_size = images.size()[0]
            
            # 真の画像を判定
            d_out_real, _, _ = D(images)

            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images, _, _ = G(input_z)
            d_out_fake, _, _ = D(fake_images)

            # 誤差を計算 => hinge version of the adversarial loss に変更
            d_loss_real = nn.ReLU()(1.0 - d_out_real).mean()
            # 誤差 d_out_realが1以上で誤差が0になる。
            # d_out_real > 1 で、1.0 - d_out_realが負の場合ReLUで0にする

            d_loss_fake = nn.ReLU()(1.0 + d_out_fake).mean()
            # 誤差 d_out_fakeが-1以下なら誤差0になる。
            # d_out_fake < -1 で、1.0 + d_out_realが負の場合ReLUで0にする

            d_loss = d_loss_real + d_loss_fake

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # ------------------
            # 2. Generatorの学習
            # ------------------
            # 偽の画僧を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images, _, _ = G(input_z)
            d_out_fake, _, _ = D(fake_images)

            # 誤差を計算 -> hinge version of the adversarial lossに変更
            g_loss = d_out_fake.mean()

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # ------------------
            # 3. 記録
            # ------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G_Loss:{:.4f}'.format(epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
        print('timer : {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    # print('総イテレーション回数 : ', iteration)

    return G, D

# ネットワークの初期化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.2)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias,data, 0)


# ファイルリストを作成
train_img_list = make_datapath_list()

# Datasetを作成
mean = (0.5,)
std = (0.5,)
transform = ImageTrandform(mean, std)
train_dataset = GAN_Img_Dataset(train_img_list, transform=transform)

# Dataloaderを作成
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 動作確認
G = Generator(z_dim=20, image_size=64)
D = Discriminator(z_dim=20, image_size=64)

# 初期化の実施
G.apply(weights_init)
D.apply(weights_init)

print('ネットワークの初期化完了')

num_epochs = 10
G_update, D_update = train_model(G, D, dataloader=train_dataloader, num_epochs=num_epochs)


