#download all negative images from:
#http://vis-www.cs.umass.edu/lfw/
#untar all images in folder lfw and put it inside data
#after move all negative images into the negative folder, remove the lfw folder

import os
from pathlib import Path

cur_dir = Path(__file__).resolve().parent
lfw_dir = os.path.join(cur_dir,'data/lfw')
pos_dir = os.path.join(cur_dir,'data/positive')
neg_dir = os.path.join(cur_dir,'data/negative')
anchor_dir = os.path.join(cur_dir,'data/anchor')
# print(lfw_dir)

for directory in [d for d in os.listdir(lfw_dir) if not d.startswith('.')]:
    print(directory)
    for file in os.listdir(os.path.join(lfw_dir, directory)):
        EX_PATH = os.path.join(lfw_dir, directory, file)
        NEW_PATH = os.path.join(neg_dir, file)
        os.replace(EX_PATH, NEW_PATH)

os.remove(lfw_dir)
