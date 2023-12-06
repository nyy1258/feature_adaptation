import json
import os
from collections import defaultdict
import shutil
import random


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

file_path = "DATA/oxford_pets/split_zhou_OxfordPets.json"

d = read_json(file_path)
name_d = defaultdict(int)
number_d = defaultdict(int)

# ['train', 'val', 'test']
print("train len: ", len(d['train']))
print("val len: ", len(d['val']))
print("test len: ", len(d['test']))

# about 80 pictures per class (0 ~ 36 total 37 classes)
for item in d['train']:
    name_d[item[1]] += 1
    number_d[item[2]] += 1

# print(name_d)
# print(number_d)


root = "DATA/OfficeHome_ori/Real World"
des_root = "DATA/OfficeHome"

# val set
folders = os.listdir(root)
classes = []
idx = 0

for f in folders:
    classes.append(f)
    dataroot = os.path.join(root, f)
    datas = os.listdir(dataroot)
    num = 0
    filenames = []
    for d in datas:
        filenames.append(d)
        num += 1
    print(f, " :", num)

    val_split = int(num*0.2)

    val = random.sample(filenames, val_split)
    #print(val)
    
    des_folder = os.path.join(des_root, "val", "{}_{}".format(idx, f))
    os.makedirs(des_folder, exist_ok=True)
    idx += 1

    for f in val:
        src = os.path.join(dataroot, f)
        des = os.path.join(des_folder, f)
        shutil.move(src, des)

print("number of classes: ", len(classes))

# train
folders = os.listdir(root)
classes = []
idx = 0

for f in folders:
    classes.append(f)
    dataroot = os.path.join(root, f)
    datas = os.listdir(dataroot)
    num = 0
    filenames = []
    for d in datas:
        filenames.append(d)
        num += 1

    train_split = int(num*0.5)

    train = random.sample(filenames, train_split)
    #print(val)
    
    des_folder = os.path.join(des_root, "train", "{}_{}".format(idx, f))
    os.makedirs(des_folder, exist_ok=True)
    idx += 1

    for f in train:
        src = os.path.join(dataroot, f)
        des = os.path.join(des_folder, f)
        shutil.move(src, des)


# test
folders = os.listdir(root)
classes = []
idx = 0

for f in folders:
    classes.append(f)
    dataroot = os.path.join(root, f)
    datas = os.listdir(dataroot)
    num = 0
    filenames = []
    for d in datas:
        filenames.append(d)
        num += 1
    
    des_folder = os.path.join(des_root, "test", "{}_{}".format(idx, f))
    os.makedirs(des_folder, exist_ok=True)
    idx += 1

    for f in filenames:
        src = os.path.join(dataroot, f)
        des = os.path.join(des_folder, f)
        shutil.move(src, des)