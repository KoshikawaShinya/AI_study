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

# 実行
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_list = make_datapath_list(phase='train')
val_list = make_datapath_list(phase='val')
print(train_list)

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
batch_size = 32

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
