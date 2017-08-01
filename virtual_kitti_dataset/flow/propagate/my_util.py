#!/usr/bin/python

from __future__ import print_function
import os
import cv2
import numpy as np
import h5py
import json


info_path = "vkitti.json"
with open(info_path, 'r') as fp:
    info = json.load(fp)
palette = np.array(info["palette"], dtype=np.uint8)


def myReMap(seg, flow_x, flow_y):
    rows, cols = seg.shape
    dst = np.ones((rows, cols), dtype=np.uint8)
    for i in xrange(rows):
        for j in xrange(cols):
            tmp_i = i+flow_y[i, j]
            tmp_j = j+flow_x[i, j]
            if tmp_i>=0 and tmp_i<=rows-1:
                tmp_i = int(round(tmp_i))
            elif tmp_i < 0:
                tmp_i = int(0)
            elif tmp_i > rows-1:
                tmp_i = int(rows-1)

            if tmp_j>=0 and tmp_j<=cols-1:
                tmp_j = int(round(tmp_j))
            elif tmp_j < 0:
                tmp_j = int(0)
            elif tmp_j > cols-1:
                tmp_j = int(cols-1)

            dst[i, j] = seg[tmp_i, tmp_j]

    return dst

def id2color(id_img):
    h, w = id_img.shape
    color_img = palette[id_img.ravel()].reshape((h, w, 3))
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    return color_img

