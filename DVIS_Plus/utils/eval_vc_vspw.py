import numpy as np
import os
from PIL import Image
# from utils import Evaluator
import sys


def get_common(list_, predlist, clip_num, h, w):
    accs = []
    for i in range(len(list_) - clip_num):
        global_common = np.ones((h, w))
        predglobal_common = np.ones((h, w))

        for j in range(1, clip_num):
            common = (list_[i] == list_[i + j])
            global_common = np.logical_and(global_common, common)
            pred_common = (predlist[i] == predlist[i + j])
            predglobal_common = np.logical_and(predglobal_common, pred_common)
        pred = (predglobal_common * global_common)

        acc = pred.sum() / global_common.sum()
        accs.append(acc)
    return accs


DIR = sys.argv[1]

Pred = sys.argv[2]
split = 'val.txt'

with open(os.path.join(DIR, split), 'r') as f:
    lines = f.readlines()
    for line in lines:
        videolist = [line[:-1] for line in lines]
total_acc = []

clip_nums = [8, 16]
for clip_num in clip_nums:
    for video in videolist:
        imglist = []
        predlist = []

        images = sorted(os.listdir(os.path.join(DIR, 'data', video, 'mask')))

        if len(images) <= clip_num:
            continue
        for imgname in images:
            img = Image.open(os.path.join(DIR, 'data', video, 'mask', imgname))
            w, h = img.size
            img = np.array(img)
            imglist.append(img)
            pred = Image.open(os.path.join(Pred, video, imgname))
            pred = np.array(pred)
            predlist.append(pred)

        accs = get_common(imglist, predlist, clip_num, h, w)
        print(sum(accs) / len(accs))
        total_acc.extend(accs)
    Acc = np.array(total_acc)
    Acc = np.nanmean(Acc)
    print(Pred)
    print('*' * 10)
    print('VC{} score: {} on {} set'.format(clip_num, Acc, split))
    print('*' * 10)