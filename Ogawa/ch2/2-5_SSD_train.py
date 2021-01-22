import os.path as osp
import random
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from utils.ssd_model import make_datapath_list, VOCDataset, DataTransform, Anno_xml2list, od_collate_fn, SSD
from utils.ssd_model import MultiBoxLoss


""" データローダーの作成 """

# ファイルパスのリストを取得
rootpath = "./data/VOCdevkit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

# Datasetを作成
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
color_mean = (104, 117, 123)    # (BGR)の色の平均値
input_size = 300    # 画像のinputサイズを300x300にする

train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train', transform=DataTransform(input_size, color_mean),
                           transform_anno=Anno_xml2list(voc_classes))
val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val', transform=DataTransform(input_size, color_mean),
                         transform_anno=Anno_xml2list(voc_classes))

# DataLoaderを作成
batch_size = 32

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
val_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)

# 辞書オブジェクトにまとめる
dataloaders_dict = {'train' : train_dataloader, 'val' : val_dataloader}


# SSD300の設定
ssd_cfg = {
    'num_classes' : 21, # 背景クラスを含めた合計クラス数
    'input_size' : 300, # 画像入力サイズ
    'bbox_aspect_num' : [4, 6, 6, 6, 4, 4], # 出力するDBoxのアスペクト比の種類
    'feature_maps' : [38, 19, 10, 5, 3, 1], # 各sourceの画像サイズ
    'steps' : [8, 16, 32, 64, 100, 300],    # DBoxの大きさを決める
    'min_sizes' : [30, 60, 111, 162, 213, 264],     # DBoxの大きさを決める
    'max_sizes' : [60, 111, 162, 213, 264, 315],    # DBoxの大きさを決める
    'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
}


""" SSDネットワークモデルの作成 """

# SSDネットワークモデル
net = SSD(phase='train', cfg=ssd_cfg)

# SSDの初期の重みを設定
# SSDのvgg部分に重みをロードする
vgg_weights = torch.load('./weights/vgg16_reducedfc.pth')
net.vgg.load_state_dict(vgg_weights)

# SSDのその他のネットワークの重みはHeの初期値で初期化

def weights_init(m):
    if isinstance(m, nn.Conv2d):    # mがnn.conv2dである場合
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)

# Heの初期値を適用
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

# GPUが使えるかを確認
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('使用デバイス : ', device)
print('ネットワーク設定完了:学習済みの重みをロードしました')


""" 損失関数と最適化手法の設定 """

# 損失関数の設定
criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

# 最適化手法の設定
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)


""" 学習と検証の実施 """

# モデルを学習させる関数を作成
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用デバイス : ', device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # イテレーションカウンタをセット
    iteration = 1
    epoch_train_loss = 0.0  # epochの損失和
    epoch_val_loss = 0.0    # epochの損失和
    logs = []

    # epochのループ
    for epoch in range(num_epochs+1):

        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() # モデルを訓練モードに
                print(' (train) ')
            else:
                if((epoch+1) % 10 == 0):
                    net.eval()  # モデルを検証モードに
                    print('-------------')
                    print(' (val) ')
                else:
                    # 検証は10回に1回だけ行う
                    continue

            # データローダーからminibatchずつ取り出すループ
            for images, targets in dataloaders_dict[phase]:

                # GPUが使えるならGPUにデータを送る
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]   # リストの各要素のテンソルをGPUへ
                
                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬 (forward) 計算
                with torch.set_grad_enabled(phase == 'train'):  # trainの時には勾配を計算する
                    # 順伝播計算 (forward) 計算
                    outputs = net(images)

                    # 損失の計算
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward() # 勾配の計算

                        # 勾配が大きくなりすぎると計算が不安定になるため、clipで最大でも勾配2.0にとどめる
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)

                        optimizer.step()    # パラメータの更新

                        if iteration % 10 == 0:     # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('イテレーション {} || Loss : {:.4f} || 10iter : {:.4f} sec.'.format(
                                iteration, loss.item(), duration))
                            t_iter_start = time.time()
                        
                        epoch_train_loss += loss.item()
                        iteration += 1

                    # 検証時
                    else:
                        epoch_val_loss += loss.item()
        
        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_train_loss : {:.4f} || Epoch_val_loss : {:.4f}'.format(epoch+1, epoch_train_loss, epoch_val_loss))
        print('timer : {:.4f}'.format(t_epoch_finish-t_epoch_start))
        t_epoch_start = time.time()
        
        # ログを保存
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv('log_output.csv')

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # ネットワークを保存する
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), 'weights/ssd300_' + str(epoch+1) + '.pth')

# 学習・検証を実行
num_epochs = 50
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)



