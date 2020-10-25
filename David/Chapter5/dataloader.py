import numpy as np
import cv2
import glob


class DataLoaderCycle_npz:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        self.dataset = np.load('datasets/%s.npz' % self.dataset_name)

        self.imgs_A = self.dataset['A_imgs']
        self.imgs_B = self.dataset['B_imgs']


    def load_batch(self, batch_size=1):

        for i in range(((self.imgs_A.shape[0]+self.imgs_B.shape[0]) / 2) / batch_size):
            idx_A = np.random.randint(0, self.imgs_A.shape[0], batch_size)
            idx_B = np.random.randint(0, self.imgs_B.shape[0], batch_size)

            imgs_A_batch = self.imgs_A[idx_A]
            imgs_B_batch = self.imgs_B[idx_B]

            yield imgs_A_batch, imgs_B_batch

class DataLoaderCycle:
    def __init__(self, dataset_name, img_shape):
        self.dataset_name = dataset_name
        self.img_shape = img_shape

    def load_batch(self, batch_size=1):

        imgA_pathes = glob.glob('imgs/%s/trainA/*'%(self.dataset_name))
        imgB_pathes = glob.glob('imgs/%s/trainB/*'%(self.dataset_name))

        if len(imgA_pathes) < len(imgA_pathes):
            length = len(imgA_pathes)
        else:
            length = len(imgB_pathes)

        for i in range(length // batch_size):
            imgA_batch = []
            imgB_batch = []

            idx_A = np.random.choice(imgA_pathes, batch_size, replace=False)
            idx_B = np.random.choice(imgB_pathes, batch_size, replace=False)
            
            for img_A, img_B in zip(idx_A, idx_B):
                img_A = cv2.imread(img_A, cv2.IMREAD_COLOR)
                img_B = cv2.imread(img_B, cv2.IMREAD_COLOR)

                img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
                img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

                img_A = cv2.resize(img_A, (self.img_shape[0], self.img_shape[1]))
                img_B = cv2.resize(img_B, (self.img_shape[0], self.img_shape[1]))

                imgA_batch.append(img_A)
                imgB_batch.append(img_B)
            
            imgA_batch = np.array(imgA_batch, dtype=np.float32) / 127.5 - 1.0
            imgB_batch = np.array(imgB_batch, dtype=np.float32) / 127.5 - 1.0
            print(imgA_batch.dtype)

            yield imgA_batch, imgB_batch
            
    def load_img(self):

        imgA_pathe = glob.glob('imgs/%s/trainA/*'%(self.dataset_name))
        imgB_pathe = glob.glob('imgs/%s/trainB/*'%(self.dataset_name))


        idx_A = np.random.choice(imgA_pathe, replace=False)
        idx_B = np.random.choice(imgB_pathe, replace=False)
            
        img_A = cv2.imread(idx_A, cv2.IMREAD_COLOR)
        img_B = cv2.imread(idx_B, cv2.IMREAD_COLOR)

        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        img_A = cv2.resize(img_A, (self.img_shape[0], self.img_shape[1]))
        img_B = cv2.resize(img_B, (self.img_shape[0], self.img_shape[1]))

        img_A = np.array(img_A) / 127.5 - 1.0
        img_B = np.array(img_B) / 127.5 - 1.0

        return img_A, img_B