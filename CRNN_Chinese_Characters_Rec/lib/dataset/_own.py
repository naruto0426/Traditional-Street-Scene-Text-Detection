from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2
from tqdm import tqdm
from autoaugment import ImageNetPolicy
import torchvision.transforms as transforms
import numpy as np
import cv2
class _OWN(data.Dataset):
    def __init__(self, config, is_train=True):

        self.roots = config.DATASET.ROOT
        if type(self.roots)!=list:
            self.roots = [self.roots]
        print(self.roots)
        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        txt_files = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']

        # convert name:indices to name:string
        self.labels = []
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            ImageNetPolicy(), 
        ])

        self.imgs = []
        if type(txt_files)!=list:
            txt_files = [txt_files]
        for i,txt_file in enumerate(txt_files):
            with open(txt_file, 'r', encoding='utf-8') as file:
            	for c in tqdm(file.readlines()):
                    path = os.path.join(self.roots[i],c.split(',')[0])
                    label = c.split(',')[-1][:-1]
                    self.labels += [{path: label}]
                    #self.imgs += [cv2.imread(path)]
        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)
    def process_img(self,img):
        img_h, img_w = img.shape

        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        return img
    def __getitem__(self, idx):
        path = list(self.labels[idx].keys())[0]
        img = cv2.imread(path)
        #img = self.imgs[idx]
        img = np.asarray(self.transform(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = img.shape
        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx








