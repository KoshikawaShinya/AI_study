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
    
