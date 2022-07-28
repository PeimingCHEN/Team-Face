#download all negative images from:
#http://vis-www.cs.umass.edu/lfw/
#untar all images in folder lfw and put it inside data
#after move all negative images into the negative folder, remove the lfw folder

import os
from pathlib import Path
import math
import shutil

cur_dir = Path(__file__).resolve().parent
lfw_dir = os.path.join(cur_dir,'data/lfw')
pos_dir = os.path.join(cur_dir,'data/positive')
neg_dir = os.path.join(cur_dir,'data/negative')
anchor_dir = os.path.join(cur_dir,'data/anchor')

# for directory in [d for d in os.listdir(lfw_dir) if not d.startswith('.')]:
#     print(directory)
#     for file in os.listdir(os.path.join(lfw_dir, directory)):
#         EX_PATH = os.path.join(lfw_dir, directory, file)
#         NEW_PATH = os.path.join(neg_dir, file)
#         os.replace(EX_PATH, NEW_PATH)


for directory in os.listdir(lfw_dir):
    imagelist = []
    for image in os.listdir(os.path.join(lfw_dir, directory)):
        imagelist.append(image)
    img_count = len(imagelist)
    if img_count >3:
        os.makedirs(os.path.join(anchor_dir, directory))
        os.makedirs(os.path.join(pos_dir, directory))
        for i in range(math.ceil(img_count/2)):
            EX_PATH = os.path.join(lfw_dir, directory, imagelist[i])
            NEW_PATH = os.path.join(anchor_dir, directory, imagelist[i])
            shutil.copyfile(EX_PATH, NEW_PATH)
        for i in range(img_count//2, img_count):
            EX_PATH = os.path.join(lfw_dir, directory, imagelist[i])
            NEW_PATH = os.path.join(pos_dir, directory, imagelist[i])
            shutil.copyfile(EX_PATH, NEW_PATH)
    else:
        for image in os.listdir(os.path.join(lfw_dir, directory)):
            EX_PATH = os.path.join(lfw_dir, directory, image)
            NEW_PATH = os.path.join(neg_dir, image)
            shutil.copyfile(EX_PATH, NEW_PATH)
            
# imagelist = []
# for directory in os.listdir(pos_dir):
#     for image in os.listdir(os.path.join(pos_dir, directory)):
#         imagelist.append(image)
# img_count = len(imagelist)
# print('pos num:',img_count)
# imagelist = []
# for directory in os.listdir(anchor_dir):
#     for image in os.listdir(os.path.join(anchor_dir, directory)):
#         imagelist.append(image)
# img_count = len(imagelist)
# print('anchor num:',img_count)