from keras.layers import Conv2D, Input, Concatenate, Dropout, Activation, UpSampling2D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import Model
from keras.optimizers import Adam

class CycleGan:

    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.channels = self.img_shape[2]

        # 生成器の最初の層のフィルタ数
        self.gen_n_filters = 32

        self.optimizer = Adam(lr=0.001)


    def build_generator_unet(self):

        def downsample(layer_input, filters, kernel_size=4):
            d = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(layer_input)
            d = InstanceNormalization(axis=-1, center=False, scale=False)(d)
            d = Activation('relu')(d)

            return d
        
        def upsample(layer_input, skip_input, filters, kernel_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(u)
            u = InstanceNormalization(axis=-1, center=False, scale=False)(u)
            u = Activation('relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            
            u = Concatenate()([u, skip_input])
            return u
        
        # 入力画像
        img = Input(shape=self.img_shape)

        # ダウンサンプリング
        d1 = downsample(layer_input=img, filters=self.gen_n_filters)
        d2 = downsample(layer_input=d1, filters=self.gen_n_filters*2)
        d3 = downsample(layer_input=d2, filters=self.gen_n_filters*4)
        d4 = downsample(layer_input=d2, filters=self.gen_n_filters*8)

        # アップサンプリング
        u1 = upsample(layer_input=d4, skip_input=d3, filters=self.gen_n_filters*4)
        u2 = upsample(layer_input=u1, skip_input=d2, filters=self.gen_n_filters*2)
        u3 = upsample(layer_input=u2, skip_input=d1, filters=self.gen_n_filters)

        u4 = UpSampling2D(size=2)(u3)
        output = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(img, output)
        


img_size = 64
img_shape = (img_size, img_size, 3)