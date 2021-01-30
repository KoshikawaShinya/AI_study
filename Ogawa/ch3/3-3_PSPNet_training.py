import torch
from torch import nn
from torch.utils import data

from utils.dataloader import make_datapath_list, DataTransform, VOCDataset
from utils.pspnet import PSPNet

# ファイルパスリスト作成
rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath=rootpath)

# Datasetの作成
# (RGB)の色の平均値と標準偏差
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)

train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train', transform=transform)
val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val', transform=transform)

# DataLoader作成
batch_size = 8

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 辞書型変数にまとめる
dataloaders_dict = {'train' : train_dataloader, 'val' : val_dataloader}

# ファインチューニングでPSPNetを作成
# ADE20Kデータセットの学習済みモデルを使用、ADE20Kはクラス数が150
net = PSPNet(n_classes=150)

# ADE20K学習済みパラメータをロード
state_dict = torch.load('./weights/pspnet50_ADE20K.pth')
net.load_state_dict(state_dict)

# 分類用の畳み込み層を、出力層21のものに付け替える
n_classes = 21
net.decode_feature.classification = nn.Conv2(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
net.aux.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

# 付け替えた畳み込み層を初期化する。活性化関数がシグモイド関数のため、Xavierを使用する

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None : # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)

net.decode_feature.classification.apply(weights_init)
net.aux.classification.apply(weights_init)

print('ネットワーク設定完了 : 学習済みの重みをロードしました')
