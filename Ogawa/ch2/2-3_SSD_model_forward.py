from math import sqrt
from itertools import product

import pandas as pd
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# オフセット情報を使い、DBoxをBBoxに変換する関数
def decode(loc, dbox_list):
    """
    オフセット情報を使い、DBoxをBBoxに変換する。

    Parameters
    ----------
    loc : [8732, 4]
        SSDモデルで推論するオフセット情報
    dbox_list : [8732, 4]
        DBoxの情報

    Returns
    -------
    boxes : [xmin, ymin, xmax, ymax]
        BBoxの情報
    """

    # DBoxは[cx, cy, width, height]で格納されている
    # locも[Δcx, Δcy, Δwidth, Δheight]で格納されている

    # オフセット情報からBBoxを求める
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],     # BBoxのcx, cyの計算        torch.Size([8732, 2])
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)     # BBoxのwidth, heightの計算 torch.Size([8732, 2])
                                                                    # torch.catでdim=1にして連結する。そのためboxesは[8732, 4]の形となる。
    
    # BBoxの座標情報を[cx, cy, width, height]から[xmin, ymin, xmax, ymax]に
    boxes[:, :2] -= boxes[:, 2:] / 2    # 座標(xmin, ymin)へ変換
    boxes[:, 2:] += boxes[:, :2]        # 座標(xmax, ymax)へ変換

    return boxes
    
# Non-Maximum Suppressionを行う関数
def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maximum Suppressionを行う関数
    boxesのうち、かぶりすぎ(overlap以上)のBBoxを削除する

    Parameters
    ----------
    boxes : [確信度閾値(0.01)を超えたBBox数, 4]
        BBox情報
    scores : [確信度閾値(0.01)を超えたBBox数]
        confの情報

    Returns
    -------
    keep : リスト
        confの降順にnmsを通過したindexが格納
    count : int
        nmsを通過したBBoxの数
    """
    
    # return のひな形を作成
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()    # 整数を扱うためlong型にする。
    # keep : torch.Size([確信度閾値を超えたBBox数])、要素は全部0

    # 各BBoxの面積areaを計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)  # 掛け算

    # boxesと同じtorch.dtypeとtorch.deviceを持つテンソルを作成
    # 後で、BBoxのかぶり度合いIOUの計算に使用するひな形として用意
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # scoreを昇順に並べる
    v, idx = scores.sort(dim=0)

    # 上位top_k個(200個)のBBoxのindexを取り出す(200個存在しない場合もある)
    idx = idx[-top_k:]

    # idxの要素が0でない限りループする
    while idx.numel() > 0:
        i = idx[-1]     # 現在のconf最大のindexをiにで移入

        # keepの現在の最後にconf最大のindexを格納
        keep[count] = i
        count += 1

        # 最後のBBoxになった場合はループを抜ける
        if idx.size(0) == 1:
            break
            
        # 現在のconf最大のindexをkeepに格納したので、idxの要素の最後を1つ減らす
        idx = idx[:-1]

        # ---------------
        # これからkeepに格納したBBoxとかぶりの大きいBBoxを抽出して除去する
        # ---------------
        # 1つ減らしたidxまでのBBoxを、outに指定した変数として作成する
        # torch.index_selectの2つめの引数dim=0となっているため、ここではtmp_x1 = x1[idx]と同じ結果となる。
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # 全てのBBoxに対して、現在のBBox=indexがiとかぶっている値までに設定(clamp)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        # wとhのテンソルサイズをindexを1つ減らしたものにする
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # clampした状態でのBBoxの幅と高さを求める
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # 幅や高さが負になっているものは0にする
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # clampされた状態での面積を求める
        inter = tmp_w * tmp_h

        # IoU = Intersect部分 / (area(a) + area(b) - intersect部分)の計算
        # IoUとは、最も高い予測値を出したBBoxとその他の1つのBBoxのエリアを選び出し、その2つのエリアの重なっている部分をIntersectと呼ぶ。
        # このIntersectを2つのBBoxの和(OR)で割ることで求まる、2つのエリアの重なり度合いである。
        rem_areas = torch.index_select(area, 0, idx)    # 各BBoxの元の面積
        union = (rem_areas - inter) + area[i]   # 2つのエリアの和(OR)の面積
        IoU = inter / union # 重なっている部分の面積 / 2つのエリアの和(OR)の面積

        # IoUがoverlapより小さいidxのみを残す
        idx = idx[IoU.le(overlap)]  # leはLess than or Equal to の処理をする演算
        # IoUがoverlapより大きいidxは、最初に選んでkeepに格納したidxと同じ物体に
        # 対してBBoxを囲んでいるため消去

    # whileのループが抜けたら終了

    return keep, count


