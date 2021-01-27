import os.path as osp
from PIL import Image

import torch.utils.data as data

def make_datapath_list(rootpath):
    """
    学習、検証の画像データとアノテーションデータへのファイルパスリストを作成

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
    annopath_template = osp.join(rootpath, 'SegmentationClass', '%s.jpg')

    # 訓練と検証、それぞれのファイルのID(ファイル名)を取得する
    train_id_names = osp.join(rootpath + 'ImageSets/Segmentation/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Segmentation/val.txt')

    # 訓練データの画像ファイルとアノテーションファイルへのパスリストを作成
    train_img_list = []
    train_anno_list = []

    for line in open(train_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id) # 画像のパス
        anno_path = (annopath_template % file_id)   # アノテーションのパス
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
    
    # 検証データの画像ファイルとアノテーションファイルへのパスリストを作成
    val_img_list = []
    val_anno_list = []

    for line in open(val_id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = (imgpath_template % file_id) # 画像へのパス
        anno_path = (annopath_template % file_id)   # アノテーションのパス
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)
    
    return train_img_list, train_anno_list, val_img_list, val_anno_list

# 動作確認
rootpath = './data/VOCdevkit/VOC2012/'
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

