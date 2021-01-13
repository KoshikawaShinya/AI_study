from itertools import product

from torch import nn
from torch.nn import init
import torch

# 34層にわたる、vggモジュールを作成
def make_vgg():
    layers = []
    in_channels = 3 # 色チャネル数

    # vggモジュールで使用する畳み込み層やマックスプーリングのチャネル数
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        elif v == 'MC':
            # ceilは出力サイズを、計算結果(float)に対して、切り上げで整数にするモード
            # デフォルトでは出力サイズを計算結果(float)に対して、切り下げで整数にする
            # floorモード
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]   # inplaceをTrueにすることにより、メモリを節約
            in_channels = v
        
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)

# 8層にわたる、extrasモジュールを作成
def make_extras():
    layers = []
    in_channels = 1024  # vggモジュールから出力された、extraに入力される画像チャネル数

    # extraモジュールの畳み込み層のチャネル数を設定するコンフィグレーション
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels=in_channels, out_channels=cfg[0], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfg[0], out_channels=cfg[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(in_channels=cfg[1], out_channels=cfg[2], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfg[2], out_channels=cfg[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(in_channels=cfg[3], out_channels=cfg[4], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfg[4], out_channels=cfg[5], kernel_size=3)]
    layers += [nn.Conv2d(in_channels=cfg[5], out_channels=cfg[6], kernel_size=1)]
    layers += [nn.Conv2d(in_channels=cfg[6], out_channels=cfg[7], kernel_size=3)]

    return nn.ModuleList(layers)

# デフォルトボックスのオフセットを出力するloc_layers
# デフォルトボックスに対する各クラスの信頼度confidenceを出力するconf_layersを作成

def make_loc_con(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):

    loc_layers = []
    conf_layers = []

    # VGGの22層目、conv4_3(source1)に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)]

    # VGGの最終層(source2)に対する畳み込み層
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)]

    # extraの(source3)に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)]

    # extraの(source4)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)]

    # extraの(source5)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)]

    # extraの(source6)に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

# conv4_3空の出力をscale=20のL2Normで正規化する層
class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()  # 親クラスのコンストラクタ実行
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters() # パラメータの初期化
        self.eps = 1e-10

    def reset_parameters(self):
        '''結合パラメータを大きさscaleの値にする初期化を実行'''
        init.constant_(self.weight, self.scale) # weightの値が全てscale(=20)になる
    
    def forward(self, x):
        '''38x38の特徴量に対して、512チャネルにわたって2乗和のルートを求めた
        38x38個の値を使用し、各特徴量を正規化してから係数を掛け算する層'''

        # 各チャネルにおける38x38個の特徴量のチャネル方向の2乗和を計算し、
        # さらにルートを求め、割り算して正規化する
        # normのテンソルサイズはtorch.Size([batch_num, 1, 38, 38])となる

        # ノルムを求める
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # 入力の各値ををノルムで割る
        x = torch.div(x, norm)

        # 係数をかける。係数はチャネルごとに1つで、512個の係数を持つ
        # self.weightのテンソルサイズはtorch.Size([512])なので
        # torch.Size([batch_num, 512, 38, 38])まで変形する

        # unsqueezeで次元を増やす
        # expand_asでxと同じ形にする
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out

# デフォルトボックスを出力するクラス
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        # 初期設定
        self.image_size = cfg['input_size'] # 画像サイズの300
        # [38, 19, ...] 各sourceの特徴量マップのサイズ
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg['features_maps']) # sourceの個数=6
        self.steps = cfg['steps']   # [8, 16, ...] DBoxのピクセルサイズ
        self.min_sizes = cfg['min_sizes']   # [30, 60, ...] 小さい正方形のDBoxのピクセルサイズ
        self.max_sizes = cfg['max_sizes']   # [60, 111, ...] 大きい正方形のDBoxのピクセルサイズ
        self.aspect_ratios = cfg['aspect_ratios']   # 長方形のDBoxのアスペクト比

    def make_dbox_list(self):
        '''DBoxを作成する'''
        mean = []
        # 'feature_maps' : [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):    # fまでの数で2ペアの組み合わせを作る 
                                                        # これにより、分割した特徴量マップの座標を総当たりする
                # 特徴量の画像サイズ
                # 300 / 'steps' : [8, 16, 32, 64, 100, 300]
                f_k = self.image_size / self.steps[k]

                # DBoxの中心座標x, y ただし、0~1で規格化している
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # アスペクト比1の小さいDBox [cx, cy, width, height]
                # 'min_sizes' : [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k] / self.image_size



 

# 動作確認
vgg_test = make_vgg()
print(vgg_test)
extras_test = make_extras()
print(extras_test)

loc_test, conf_test = make_loc_con()
print(loc_test)
print(conf_test)