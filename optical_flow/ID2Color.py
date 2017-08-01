#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import cv2
import json
import numpy as np
from os.path import dirname, exists, join, splitext
import os
import glob


def process(method, down, data_type):

    cwd = os.getcwd()
    pattern = join(cwd, "seg", method, "down-"+str(down), data_type, "*ID.png")
    #pattern = join(cwd, "groundTruth", "down-"+str(down), data_type, "*Ids.png")
    file_list = glob.glob(pattern)
    file_list.sort()

    info_path = "cityscapes.json"
    with open(info_path, 'r') as fp:
        info = json.load(fp)
    palette = np.array(info['palette'], dtype=np.uint8)
    #black = np.array([[0, 0, 0]], dtype=np.uint8)
    #palette = np.concatenate((palette, black), axis=0)
    cnt = 0;
    for f in file_list:
        imgID = cv2.imread(f, 0)
        #imgID[imgID==255] = 19
        h, w = imgID.shape
        color_img = palette[imgID.ravel()].reshape((h, w, 3))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        img_name = f.split('/')[-1]
        output_path = join(cwd, "seg",  method, "down-"+str(down), data_type, splitext(img_name)[0][:-2]+'Color.png')
        #output_path = join(cwd, "groundTruth", "down-"+str(down), data_type, splitext(img_name)[0][:-3]+'Color.png')
        print('Writing', output_path)
        cv2.imwrite(output_path, color_img)
        if cnt == 9:
            break
        cnt += 1
        #break



if __name__ == '__main__':

    process("dilation10", 1, "train")
    process("dilation10", 1, "val")
    process("dilation10", 2, "train")
    process("dilation10", 2, "val")

