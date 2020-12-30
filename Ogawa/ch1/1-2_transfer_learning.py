import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# 入力画像の前処理をするクラス
# 訓練時と推論時で処理が異なる

class ImageTransform():
    """
    画像の前処理クラス。訓練時、検証時で異なる動作をする。
    画像のサイズをリサイズし、色を標準化する。
    訓練時はRandomResizedCropとRandomHrizontalFlipでデータおーぎゅめんテーションする。

    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ
    mean : (R, G, B)
        各色チャネルの平均値
    std : (R, G, B)
        各色チャネルの標準偏差
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(1.5, 1.0)),  # データオーギュメンテーション
                transforms.RandomHorizontalFlip(),  # データオーギュメンテーション
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std) # 標準化
            ]),
            'val' : transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.CenterCrop(resize),  # 画面中央をresize x resizeで切り取る
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std) # 標準化
            ])
        }
    
    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理モードを指定
        """
        return self.data_transform[phase](img)

# アリと八の画像のDatasetを作成する
class HymenopteraDataset(data.Dataset):
    """
    アリとハチの画像のDatasetクラス。PyTorchのDatasetクラスを継承

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'val'
        学習か訓練かを設定
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list # ファイルパスのリスト
        self.transform = transform # 前処理クラスのインスタンス
        self.phase = phase # train or val の設定

    def __len__(self):
        """画像の枚数を返す"""
        return len(self.file_list)
    
    def __getitem__(self, index):
        '''
        前処理した画像のTensor形式のデータをラベルを取得
        '''

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)   # torch.Size([3, 224, 224])

        # 画像のラベルをファイル名から抜き出す
        if self.phase == 'train':
            label = img_path[30:34]
        elif self.phase == 'val':
            label = img_path[28:32]
        
        # ラベルを数値に変更する
        if label == 'ants':
            label = 0
        elif label == 'bees' :
            label = 1
        
        return img_transformed, label


# アリと八の画像へのファイルパスのリストを作成
def make_datapath_list(phase='train'):
    """
    データのパスを格納したリストを作成

    Parameters
    ----------
    phase : 'train' or 'val'
        訓練データか検証データかを指定
    
    Returns
    -------
    path_list : list
        データへのパスを格納したリスト
    """

    # カレントディレクトリをch1とする必要がある
    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    print(target_path)

    path_list = []  # ここに格納する

    # globを利用してサブディレクトリまでファイルパスを取得
    for path in glob.glob(target_path):
        path_list.append(path)
    
    return path_list

# モデルを学習させるための関数
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの学習と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() # モデルを学習モードに
            else:
                net.eval()  # モデルを検証モードに

            epoch_loss = 0.0    # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝播(forward)計算
                # 学習時のみ勾配を計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)   # 損失を計算
                    _, preds = torch.max(outputs, 1)     # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    # イテレーション結果の計算
                    # lossの合計を更新
                    # lossにはミニバッチサイズで平均した損失が格納されている。それをinputs.size(0)のミニバッチサイズを掛けて足すことで
                    # 合計損失を求める。
                    epoch_loss += loss.item() * inputs.size(0)
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)
            
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))



# 実行
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_list = make_datapath_list(phase='train')
val_list = make_datapath_list(phase='val')

"""データセット"""
train_dataset = HymenopteraDataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = HymenopteraDataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

# 動作確認
index = 0
image, label = train_dataset.__getitem__(index)
print(image.size())
print(label)

"""データローダー"""
# ミニバッチのサイズ指定
batch_size = 16

# DataLoaderを作成
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 辞書型変数にまとめる
dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

# 動作確認
# イテレータにすることでnextにより一番目の要素を取得する
batch_iterator = iter(dataloaders_dict['train'])    # イテレータに変換
inputs, labels = next(batch_iterator)   # 一番目の要素を取り出す
print(inputs.size())
print(labels)

"""VGG-16"""
# 学習済みのVGG-16モデルをロード
# VGG-16モデルのインスタンスを生成
use_pretrained = True   # 学習済みのパラメータを使用
net = models.vgg16(pretrained=use_pretrained)

# VGG-16の最後の出力層の出力ユニットをアリとハチの２つに付け替える
net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

# 訓練モードに設定
net.train()
print('ネットワーク設定完了')

# 損失関数の設定
criterion = nn.CrossEntropyLoss()

# 転移学習で学習させるパラメータを、変数params_to_updateに格納する
params_to_update = []

# 学習させるパラメータ名
update_param_names = ['classifier.6.weight', 'classifier.6.bias']

# 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
# requires_gradをTrueにすると学習時に値が更新される
for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        print(name)
    else:
        param.requires_grad = False

# params_to_updateの中身を確認
print('-----------')
print(params_to_update)

# 最適化手法の設定
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

# 学習・検証
num_epochs=3
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)