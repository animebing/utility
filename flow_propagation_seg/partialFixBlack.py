#!/usr/bin/python

from __future__ import print_function
import os
import cv2
import numpy as np
import json
import argparse
from fixBlack import fixBlack

info_path = "/home/bingbing/git/dilation/datasets/cityscapes.json"
with open(info_path, 'r') as fp:
    info = json.load(fp)
palette = np.array(info["palette"], dtype=np.uint8)
black = np.array([[0, 0, 0]], dtype=np.uint8)
palette = np.concatenate([palette, black], axis=0)


def myReMap(seg, flow_x, flow_y):
    rows, cols = seg.shape
    dst = 19*np.ones((rows, cols), dtype=np.uint8)
    for i in xrange(rows):
        for j in xrange(cols):

            dst_i = i+flow_y[i, j]
            dst_j = j+flow_x[i, j]

            if dst_i >= 0 and dst_i <= rows-1:
                dst_i = int(round(dst_i))
            else:
                continue

            if dst_j >= 0 and dst_j <= cols-1:
                dst_j = int(round(dst_j))
            else:
                continue

            dst[dst_i, dst_j] = seg[i, j]

    return dst

def id2color(id_img):
    h, w = id_img.shape
    color_img = palette[id_img.ravel()].reshape((h, w, 3))
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    return color_img


def process(interval):

    root_dir = "/home/bingbing/Documents/flowProSeg"
    cur_dir = os.path.join(root_dir, "interval_"+str(interval), "cur_segK2I")
    black_list_file = os.path.join(cur_dir, "val_black.lst")
    black_list = open(black_list_file, 'r').readlines()

    for i in xrange(len(black_list)):

            img_name = black_list[i].strip('\n')
            black_img = cv2.imread(img_name, 0)

            partial_black = fixBlack(black_img)


            id_path = img_name[:-9]+"PartBlack.png"
            color_path = img_name[:-11]+"PartBlackColor.png"

            color_img = id2color(partial_black)
            print("Writing: ", id_path)
            cv2.imwrite(id_path, partial_black)
            print("Writing: ", color_path)
            cv2.imwrite(color_path, color_img)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval" , type=int, required=True, help="the interval between two frames")
    args = parser.parse_args()
    process(args.interval)
