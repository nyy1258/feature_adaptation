import os
import math
import random
from collections import defaultdict

import torchvision.transforms as transforms

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader

template = ['a photo of a {}, a type of product.']

class OfficeHome(DatasetBase):

    dataset_dir = 'OfficeHome'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        train, val, test = self.read_split(self.dataset_dir)
        #train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        self.template = template

        super().__init__(train_x=train, val=val, test=test)
    
    @staticmethod
    def read_split(filepath):
        def _convert(stage):
            out = []
            folder_path = os.path.join(filepath, stage)
            folders = os.listdir(folder_path)

            for f in folders:
                data_path = os.path.join(folder_path, f)
                label = int(f.split('_')[0])
                classname = f.split('_')[-1]

                datas = os.listdir(data_path)
                for d in datas:
                    impath = os.path.join(data_path, d)
                    item = Datum(
                        impath=impath,
                        label=int(label),
                        classname=classname
                    )
                    out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        train = _convert('train')
        val = _convert('val')
        test = _convert('test')

        return train, val, test