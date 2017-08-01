#!/usr/bin/python

from __future__ import print_function
import os
import cv2
import numpy as np
import h5py
import json
import sys


info_path = "vkitti.json"
with open(info_path, 'r') as fp:
    info = json.load(fp)
palette = np.array(info["palette"], dtype=np.uint8)
#black = np.array([[0, 0, 0]], dtype=np.uint8)
#palette = np.concatenate([palette, black], axis=0)



def id2color(id_img):
    h, w = id_img.shape
    color_img = palette[id_img.ravel()].reshape((h, w, 3))
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    return color_img


img_path = sys.argv[1]

id_img = cv2.imread(img_path, 0)
color_img = id2color(id_img)
cv2.imwrite("tmp_color.png", color_img)

