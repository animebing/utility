#!/usr/bin/python

from __future__ import print_function
import os
import sys
import cv2
import numpy as np
import json

info_path = "/home/bingbing/git/dilation/datasets/cityscapes.json"
with open(info_path, 'r') as fp:
    info = json.load(fp)
palette = np.array(info["palette"], dtype=np.uint8)
black = np.array([[0, 0, 0]], dtype=np.uint8)
palette = np.concatenate([palette, black], axis=0)



def id2color(id_img):
    h, w = id_img.shape
    color_img = palette[id_img.ravel()].reshape((h, w, 3))
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    return color_img




if __name__ ==  "__main__":

    img_id = cv2.imread(sys.argv[1], 0)
    img_color = id2color(img_id)

    img_name = sys.argv[1].split('/')[-1]
    path = img_name
    # path = img_name[:-6]+"Color.png"
    print("Writing: ", path)
    cv2.imwrite(path, img_color)
