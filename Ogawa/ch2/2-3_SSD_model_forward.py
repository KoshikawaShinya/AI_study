from math import sqrt
from itertools import product

import pandas as pd
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from SSD_modal import make_vgg, make_extras, make_loc_con, L2Norm, DBox


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

# SSDの推論時にconfとlocの出力から、かぶりを除去したBBoxを出力する
class Detect(Function):

    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)   # confをソフトマックス関数で正規化するために用意　confは[butch_num, 8732, 21]のため、dim=-1にすることでDBoxごとクラスのの確率が求まる
        self.conf_thresh = conf_thresh      # confがconf_thresh=0.01より高いDBoxのみを扱う
        self.top_k = top_k                  # nm_supressionでconfの高いtop_k=200個を計算に使用する
        self.nms_thresh = nms_thresh        # nm_supressionでIOUがnms_thresh=0.45より大きいと同一物体へのBBoxとみなす

    def forward(self, loc_data, conf_data, dbox_list):
        """
        順伝播の計算を実行する

        Parameters
        ----------
        loc_data : [batch_num, 8732, 4]
            オフセット情報
        conf_data : [batch_num, 8732, 21]
            検出の確信度
        dbox_list : [8732, 4]
            DBoxの情報
        
        Returns
        -------
        output : torch.Size([batch_num, 21, 200, 5])
            (batch_num, クラス, confのtop200, BBoxの情報)
        """

        # 各サイズを取得
        num_batch = loc_data.size(0)    # ミニバッチのサイズ
        num_dbox = loc_data.size(1)     # DBoxの数 = 8732
        num_classes = conf_data.size(2) # クラス数 = 21

        # confはソフトマックスを適用して正規化する
        conf_data = self.softmax(conf_data)

        # 出力の型を作成する。テンソルサイズは[minibatch数, 21, 200, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # conf_dataを[batch_num, 8732, num_classes]から[batch_num, num_classes, 8732]に順番変更
        conf_preds = conf_data.transpose(2, 1)

        # ミニバッチごとのループ
        for i in range(num_batch):

            # 1. locとDBoxから修正したBBox[xmin, ymin, xmax, ymax]を求める
            decoded_boxes = decode(loc_data[i], dbox_list)

            # confのコピーを作成
            conf_scores = conf_preds[i].clone()

            # 画像クラスごとのループ(背景クラスのindexである0は計算せず、index=1から)
            for cl in range(1, num_classes):

                # 2. confの閾値を超えたBBoxを取り出す
                # confの閾値を超えているかのマスクを作成し、
                # 閾値を超えたconfのインデックスをc_maskとして取得
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # gtはGreater thanのこと。gtにより閾値を超えたものが1に、以下が0になる
                # conf_scores:torch.Size([21, 8732])
                # c_mask:torch.Size([8732])

                # scoresはtorch.Size([閾値を超えたBBox数])
                # c_maskでconf_scoresをフィルタリングする
                scores = conf_scores[cl][c_mask]

                # 閾値を超えたconfがない場合、つまりscores=[]の時はなにもしない
                if scores.nelement() == 0:  # nelementでそう素数の合計を求める
                    continue

                # c_maskを、decoded_boxesに適用するようにサイズを変更
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask:torch.Size([8732, 4])

                # l_maskを、decoded_boxesに適応
                boxes = decoded_boxes[l_mask].view[-1, 4]
                # decoded_boxes[l_mask]で1次元になってしまうため、
                # viewで(閾値を超えたBBox数, 4)サイズに変形しなおす

                # 3. Non-Maximum Suppressionを実施し、かぶっているBBoxを取り除く
                ids, count = nm_suppression(boxes, scores, self.nms_thresh, self.top_k)
                # ids : confの降順にNon-Suppressionを通過したindexが格納
                # count : Non-Maximum Suppressionを通過したBBoxの数

                # outputにNon-Maximum Suppressionを抜けた結果を格納
                output[i, ck, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output   # torch.Size([1, 21, 200, 5])

# SSDクラスを作成
class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase
        self.num_classes = cfg['num_classes']   # クラス数=21

        # SSDのネットワークを作る
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_con(cfg['num_classes'], cfg['bbox_aspect_num'])

        # DBox作成
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list

        # 推論時はクラス「Detext」を用意する
        if phase == 'inference':
            self.detect = Detect()
        
    def forward(self, x):
        sources = list() # locとconfへの入力source1~6を格納
        loc = list()    # locの出力を格納
        conf = list()   # confの出力を格納

        # vggのconv4_3まで計算する
        for k in range(23):
            x = self.vgg[k](x)
        
        # conv4_3の出力をL2Normに入力し、source1を作成、sourcesに追加
        source1 = self.L2Norm(x)
        sources.append(source1)

        # vggを最後まで計算し、source2を作成、sourcesに追加
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        # extrasのconvとReLUを計算
        # source3~6を、sourcesに追加
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)  # inplace=Trueにすることでメモリ節約
            if k % 2 == 1:  # conv -> ReLU -> conv -> Reluをしたらsourceにいれる
                sources.append(x)

        # source1~6に、それぞれ対応する畳み込みを1回ずつ適用する
        # zipでforループの複数のリストの要素を取得
        # source1~6まであるので、6回ループが回る
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # Permuteは要素の順番を入れ替え



