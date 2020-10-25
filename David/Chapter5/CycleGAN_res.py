from keras.layers import Conv2D, Conv2DTranspose, Input, Concatenate, Dropout, Activation, UpSampling2D, BatchNormalization, Add
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

from dataloader import DataLoaderCycle


class CycleGan:

    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.channels = self.img_shape[2]

        # データローダー
        self.dataloader = DataLoaderCycle('apple2orange', self.img_shape)

        # 生成器の最初の層のフィルタ数
        self.gen_n_filters = 32
        # 識別器の最初の層のフィルタ数
        self.disc_n_filters = 32

        self.optimizer = Adam(lr=0.0002, beta_1=0.5)

        # 統合モデルの損失の重み
        self.lambda_valid = 1
        self.lambda_reconstr = 10
        self.lambda_id = 5


        # 識別器
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()

        self.d_A.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])
        self.d_B.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])

        # 生成器
        self.g_AB = self.build_generator_resnet()
        self.g_BA = self.build_generator_resnet()

        self.d_A.trainable = False
        self.d_B.trainable = False

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        fake_A = self.g_BA(img_B)
        fake_B = self.g_AB(img_A)
        # 敵対性
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        # サイクル一貫性
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # 同一性
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[self.lambda_valid, self.lambda_valid, self.lambda_reconstr, self.lambda_reconstr, self.lambda_id, self.lambda_id],
                              optimizer=self.optimizer)


    def build_generator_unet(self):

        def downsample(layer_input, filters, kernel_size=4):
            d = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(layer_input)
            d = InstanceNormalization(axis=-1, center=False, scale=False)(d)
            d = Activation('relu')(d)

            return d
        
        def upsample(layer_input, skip_input, filters, kernel_size=4, dropout_rate=0):
            u = UpSampling2D()(layer_input)
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
        d4 = downsample(layer_input=d3, filters=self.gen_n_filters*8)

        # アップサンプリング
        u1 = upsample(layer_input=d4, skip_input=d3, filters=self.gen_n_filters*4)
        u2 = upsample(layer_input=u1, skip_input=d2, filters=self.gen_n_filters*2)
        u3 = upsample(layer_input=u2, skip_input=d1, filters=self.gen_n_filters)

        u4 = UpSampling2D(size=2)(u3)
        output = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(img, output)

    def build_generator_resnet(self):

        def downsample(layer_input, filters, kernel_size=4):
            d = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(layer_input)
            d = InstanceNormalization(axis=-1, center=False, scale=False)(d)
            d = Activation('relu')(d)

            return d

        def upsample(layer_input, filters, kernel_size=4, dropout_rate=0):
            u = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding='same')(layer_input)
            u = InstanceNormalization(axis=-1, center=False, scale=False)(u)
            u = Activation('relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            
            return u

        def residual_block(layer_input, kernel_size, filters):
            shortcut = layer_input

            y = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(layer_input)
            y = InstanceNormalization()(y)
            y = Activation('relu')(y)

            y = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(y)
            y = InstanceNormalization()(y)

            y = Add()([y, shortcut])

            return y

        # 入力画像
        img = Input(shape=self.img_shape)

        # ダウンサンプリング
        d = Conv2D(filters=self.gen_n_filters, kernel_size=7, strides=1, padding='same')(img)
        d = downsample(layer_input=d, filters=self.gen_n_filters, kernel_size=3)
        d = downsample(layer_input=d, filters=self.gen_n_filters*2, kernel_size=3)

        # 残差ブロック
        res = residual_block(layer_input=d, filters=self.gen_n_filters*2, kernel_size=3)
        for i in range(8):
            res = residual_block(layer_input=res, filters=self.gen_n_filters*2, kernel_size=3)
        
        # アップサンプリング
        u = upsample(layer_input=res, filters=self.gen_n_filters*2, kernel_size=3)
        u = upsample(layer_input=u, filters=self.gen_n_filters, kernel_size=3)
        output = Conv2DTranspose(filters=3, kernel_size=7, strides=1, padding='same', activation='tanh')(u)
        
        model = Model(img, output)
        model.summary()

        return model

    
    def build_discriminator(self):

        def conv4(layer_input, filters, stride=2, norm=True):
            y = Conv2D(filters, kernel_size=4, strides=stride, padding='same')(layer_input)
            if norm:
                y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = LeakyReLU(alpha=0.2)(y)

            return y
        
        img = Input(shape=self.img_shape)

        y = conv4(img, filters=self.disc_n_filters, stride=2, norm=False)
        y = conv4(y, filters=self.disc_n_filters*2, stride=2)
        y = conv4(y, filters=self.disc_n_filters*4, stride=2)
        y = conv4(y, filters=self.disc_n_filters*8, stride=2)

        output = Conv2D(1, kernel_size=4, strides=1, padding='same')(y)

        return Model(img, output)

    def train(self, batch_size, epochs, sample_interval):
        patch = int(self.img_shape[0] / 2**4)
        self.disc_patch = (patch, patch, 1)

        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.dataloader.load_batch(batch_size)):
                
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_B.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])
            
            if epoch & sample_interval == 0:
                sample_images(epoch)

    
    def sample_images(self, epoch):
        r, c = 2, 3

        img_A, img_B = self.dataloader.load_img()
        
        # Translate images to the other domain
        fake_B = self.g_AB.predict(img_A)
        fake_A = self.g_BA.predict(img_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([img_A, fake_B, reconstr_A, img_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("plt_imgs/%s/%d.png" % (self.dataset_name, epoch))
        #plt.show()



img_size = 128
img_shape = (img_size, img_size, 3)
epochs = 100
batch_size = 1
sample_interval = 10

cyclegan = CycleGan(img_shape)
cyclegan.train(batch_size, epochs, sample_interval)