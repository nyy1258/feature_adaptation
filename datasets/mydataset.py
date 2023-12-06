import os
import math
import random
from collections import defaultdict

import torchvision.transforms as transforms

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from sklearn.model_selection import StratifiedKFold

template = ['a photo of a {}, a type of images.']

class Mydataset(DatasetBase):

    dataset_dir = 'ksdd2'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.files = os.listdir(self.dataset_dir)
        self.allfile = []
        self.gt = []
        self.train_set = []
        self.val_set = []
        self.train_img = []
        self.val_img = []
        self.test_img = []


        train, val, test = self.read_split(self.dataset_dir)
        #train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        self.template = template


        for file in self.files:
            if file.endswith(".png"):
                self.allfile.append(file)
                self.gt.append(int(file.split("_")[0]))



        skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

        for train_idx, val_idx in skf.split(self.allfile, self.gt):
    
            for i in train_idx:
                self.train_set.append(self.allfile[i])
            
            for i in val_idx:
                self.val_set.append(self.allfile[i])

        
        for file in self.train_set:
            impath = os.path.join(self.dataset_dir, file)
            label = file.split("_")[0]
            classname = file.split("_")[0]

            item = Datum(
                impath=impath,
                label=int(label),
                classname=classname
            )
            self.train_img.append(item)


            
        for file in self.val_set:
            impath = os.path.join(self.dataset_dir, file)
            label = file.split("_")[0]
            classname = file.split("_")[0]

            item = Datum(
                impath=impath,
                label=int(label),
                classname=classname
            )
            self.val_img.append(item)
        
        datas = os.path.join(self.dataset_dir, "testing")
        test_files = os.listdir(datas)
        for file in test_files:
            if file.endswith(".png"):
                impath = os.path.join(datas, file)
                label = file.split("_")[0]
                classname = file.split("_")[0]

                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
                self.test_img.append(item)
        
        #print(f'Reading split from {filepath}')


        super().__init__(train_x=self.train_img, val=self.val_img, test=self.test_img)
    
    @staticmethod
    def read_split(dataset_dir):
        
        train_set = []
        val_set = []
        test_set = []

        train_img = []
        val_img = []
        test_img = []

        files = os.listdir(dataset_dir)
        allfile = []
        gt = []

        for file in files:
            if file.endswith(".png"):
                allfile.append(file)
                gt.append(int(file.split("_")[0]))



        skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

        for train_idx, val_idx in skf.split(allfile, gt):
    
            for i in train_idx:
                train_set.append(allfile[i])
            
            for i in val_idx:
                val_set.append(allfile[i])

        
        for file in train_set:
            impath = os.path.join(dataset_dir, file)
            label = file.split("_")[0]
            classname = file.split("_")[0]

            item = Datum(
                impath=impath,
                label=int(label),
                classname=classname
            )
            train_img.append(item)


            
        for file in val_set:
            impath = os.path.join(dataset_dir, file)
            label = file.split("_")[0]
            classname = file.split("_")[0]

            item = Datum(
                impath=impath,
                label=int(label),
                classname=classname
            )
            val_img.append(item)
        
        datas = os.path.join(dataset_dir, "testing")
        test_files = os.listdir(datas)
        for file in test_files:
            if file.endswith(".png"):
                impath = os.path.join(datas, file)
                label = file.split("_")[0]
                classname = file.split("_")[0]

                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
                test_img.append(item)
        
        #print(f'Reading split from {filepath}')
                files = os.listdir(dataset_dir)
        allfile = []
        gt = []

        for file in files:
            if file.endswith(".png"):
                allfile.append(file)
                gt.append(int(file.split("_")[0]))



        return train_img, val_img, test_img