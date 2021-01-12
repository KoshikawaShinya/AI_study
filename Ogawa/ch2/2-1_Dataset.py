import os.path as osp
import random
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop
from utils.data_augumentation import RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMean

# 学習、検証の画像データとアノテーションデータへのファイルパスリストを作成
def make_datapath_list(rootpath):
    """
    データへのパスを格納したリストを作成

    Parameters
    ----------
    rootpath : str
        データフォルダへのパス

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        データへのパスを格納したリスト
    """

    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)    # 画像のパス
        anno_path = (annopath_template % file_id)   # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
    
    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id)     # 画像のパス
        anno_path = (annopath_template % file_id)   # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


# 「XML形式のアノテーション」を、リスト形式に変換するクラス
class Anno_xml2list(object):
    """
    一枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換

    Attributes
    ----------
    classes : リスト
        VOCのクラス名を格納したリスト
    """

    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height):
        """
        1枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト化する

        Parameters
        ----------
        xml_path : str
            xmlファイルへのパス
        width : int
            対象画像の幅
        height : int
            対象画像の高さ
    
        Returns
        -------
        ret : [[xmin, ymin, xmax, ymax, label_int], ... ]
            物体のアノテーションデータを格納したリスト。画像内に存在する物体数分だけ要素を持つ
        """

        # 画像内の全ての物体のアノテーションをこのリストに格納
        ret = []

        # xmlファイルを読み込む
        xml = ET.parse(xml_path).getroot()

        # 画像内にある物体(object)の数だけループする
        for obj in xml.iter('object'):

            # アノテーションで検知がdifficultに設定されているものは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # 1つの物体に対するアノテーションを格納するリスト
            bndbox = []

            name = obj.find('name').text.lower().strip()    # 物体名 (小文字にして、改行と空白文字を除去している)
            bbox = obj.find('bndbox')   # バウンディングボックスの情報

            # アノテーションのxmin, ymin, xmax, ymaxを取得し、0 ~ 1に規格化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # VOCは原点が(1, 1)なので1を引き算して(0, 0)に
                cur_pixel = int(bbox.find(pt).text) - 1

                # 幅、高さで規格化
                if pt == 'xmin' or pt == 'xmax':    # x方向の時は幅で除算
                    cur_pixel /= width
                else :                              # y方向の時は高さで除算
                    cur_pixel /= height
                    
                bndbox.append(cur_pixel)
            
            # アノテーションのクラス名のindexを取得して追加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            # resに[xmin, ymin, xmax, ymax, label_idx]を足す
            ret += bndbox

        return np.array(ret)

# 入力画像を前処理するクラス
class DataTransform():
    """
    画像とアノテーションの前処理クラス。訓練と推論で異なる動作をする。
    画像のサイズを300x300にする。
    学習時はデータオーギュメンテーションする。

    Attributes
    ----------
    input_size : int
        リサイズ先の画像の大きさ
    color_mean : (B, G, R)
        各色チャネルの平均
    """

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train' : Compose([
                ConvertFromInts(),      # intをfloat32に変換
                ToAbsoluteCoords(),     # アノテーションデータの規格化を戻す
                PhotometricDistort(),   # 画像の色調などをランダムに変化
                Expand(color_mean),     # 画像のキャンバスを広げる
                RandomSampleCrop(),     # 画像内の部分をランダムに抜き出す
                RandomMirror(),         # 画像を反転させる
                ToPercentCoords(),      # アノテーションデータを0-1に規格化
                Resize(input_size),     # 画像サイズをinput_size x input_sizeに変形
                SubtractMean(color_mean)# BGRの色の平均値を引き算
            ]),
            'val' : Compose([
                ConvertFromInts(),      # intをfloat32に変換
                Resize(input_size),     # 画像サイズをinput_size x input_sizeに変形
                SubtractMean(color_mean)# BGRの色の平均値を引き算
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理モードを指定
        """
        return self.data_transform[phase](img, boxes, labels)

# VOC2012のDatasetを作成
class VOCDataset(data.Dataset):
    """
    VOC2012のDatasetを作成するクラス。PyTorchのDatasetクラスを継承

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'val'
        学習か訓練かを設定する
    transform : object
        前処理クラスのインスタンス
    transform_anno : object
        xmlのアノテーションをリストに変換するインスタンス
    """

    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase  # train若しくはvalを指定
        self.transform = transform  # 画像の変形
        self.transform_anno = transform_anno    # アノテーションデータをxmlからリストへ

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のテンソル形式のデータをアノテーションを取得
        '''
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        '''
        前処理をした画像のテンソル形式のデータ、アノテーション、画像の高さ、幅を取得する
        '''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)   # [高さ][幅][色BGR]
        height, width, channels = img.shape # 画像のサイズを取得

        # 2. xml形式のアノテーション情報をリストに
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        # 3. 前処理を実施
        img, boxes, labels = self.transform(img, self.phase, anno_list[:, :4], anno_list[:, 4])
        # 色チャネルの順番がBGRになっているので、RGBに順番変更
        # さらに（高さ、幅、色チャネル）の順を（色チャネル、高さ、幅）に変換
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # BBoxとラベルをセットにしたnp.arrayを作成、変数名gtがground truth(答え)の略称
        gt = np.hstach((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width
        
# DataLoaderを作成
def od_collate_fn(batch):
    """
    Datasetから取り出すアノテーションデータのサイズが画像ごとに異なる
    画像内の物体数が2個であれば(2, 5)というサイズだが、3個であれば(3, 5)など変化する
    この変化に対応したDataLoaderを作成するためにカスタマイズした、Collate_fnを作成する
    collate_fnは、PyTorchでリストからmini-batchを作成する関数である
    ミニバッチ分の画像が並んでいるリスト変数batchに、
    ミニバッチ番号を指定する次元を先頭に1つ追加して、リストの形を変形する
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0]は画像img
        targets.append(torch.FloatTensor(sample[1]))    #sample[1]はアノテーションgt

    # imgsはミニバッチサイズのリストになっている
    # リストの要素はtorch.Size([3, 300, 300])
    # このリストをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換
    # torch.stack(imgs, dim=0)により、リストの中に3次元テンソルが格納されている形ではなく、4次元テンソルの変数とする。
    imgs = torch.stack(imgs, dim=0)

    # targetsはアノテーションデータの正解であるgtのリスト
    # リストのサイズはミニバッチサイズ
    # リストtargetsの要素は[n, 5]となっている
    # nは画像ごとに異なり、画像内にある物体の数となる
    # 5は[xmin, ymin, xmax, ymax, class_index]
    
    return imgs, targets

# ファイルパスのリストを作成
rootpath = "./data/VOCdeckit/VOC2012/"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

# 動作確認
print(train_img_list[0])

# Anno_xml2listの動作確認
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

transform_anno = Anno_xml2list(voc_classes)

# 画像の読み込み OpenCVを使用
ind = 1
image_file_path = val_img_list[ind]
img = cv2.imread(image_file_path)   # [高さ][幅][色BGR]
height, width, channels = img.shape

# アノテーションをリストで表示
print(transform_anno(val_anno_list[ind], width, height))

# DataTransformまでの動作確認

# 1. 画像読み込み
image_file_path = train_img_list[0]
img = cv2.imread(image_file_path)   # [高さ][幅][色BGR]
height, width, channels = img.shape # 画像のサイズを取得

# 2. アノテーションをリストに
transform_anno = Anno_xml2list(voc_classes)
anno_list = transform_anno(train_anno_list[0], width, height)

# 3. 元画像の表示
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 4. 前処理クラスの作成
color_mean = (104, 117, 123)    # (B, G, R)の色の平均値
input_size = 300                # 画像のinputサイズを300x300にする
transform = DataTransform(input_size, color_mean)

# 5. train画像の表示
phase = "train"
# anno_lisr[:, :4]はBBoxの位置座標。anno_list[:, 4]は物体のクラス名に対応したインデックス情報
img_transformed, boxes, labels = transform(img, phase, anno_list[:, :4], anno_list[:, 4])
plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
plt.show()


# Datasetの動作確認
color_mean = (104, 117, 123)
input_size = 300

train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train',
                           trainsform=DataTransform(input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val', trainsform=DataTransform(input_size, color_mean), 
                         transform_anno=Anno_xml2list(voc_classes))

# データの取り出し例
print(val_dataset.__getitem__(1))


# データローダーの作成
batch_size = 4

train_dataloader = data.Dataloader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)

val_dataloader = data.Dataloader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)

# 辞書型変数にまとめる
dataloaders_dict = {'train' : train_dataloader, 'val': val_dataloader}

# 動作の確認
batch_iterator = iter(dataloaders_dict['val'])  # イテレータに変換
images, targets = next(batch_iterator)  # 1番目の要素を取り出す
print(images.size())
print(len(targets))
print(targets[1].size())    # ミニバッチのサイズのリスト、各要素は[n, 5]、nは物体数