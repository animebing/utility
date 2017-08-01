#!/usr/bin/python

from __future__ import print_function
import os
import cv2
import numpy as np
import h5py
import json
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


def process():

    root_dir = "/home/bingbing/Documents/flowProSeg/interval_5"
    train_val = "val"
    flow_dir = os.path.join(root_dir, "spynet_flowK2I", train_val)

    prev_seg_dir = os.path.join(root_dir, "prev_seg", train_val)
    cur_seg_dir = os.path.join(root_dir, "cur_segK2I", train_val)
    if not os.path.isdir(cur_seg_dir):
        os.makedirs(cur_seg_dir)

    flow_list = os.listdir(flow_dir)
    flow_list.sort()
    prev_seg_list = os.listdir(prev_seg_dir)
    prev_seg_list.sort()

    adj_frame_file = "/media/data/bingbing/leftImg8bit_sequence/" + train_val + "_interval_5.lst"
    adj_frame_list = open(adj_frame_file, 'r').readlines()

    for i in xrange(len(flow_list)):
            prev_seg_file = os.path.join(prev_seg_dir, prev_seg_list[i].strip('\n'))
            prev_seg = cv2.imread(prev_seg_file, 0)

            flow_file = os.path.join(flow_dir, flow_list[i].strip('\n'))
            fin = h5py.File(flow_file, 'r')
            flow = np.array(fin["optical_flow"])
            flow_x = flow[0, 0, :, :]
            flow_y = flow[0, 1, :, :]
            cur_seg = myReMap(prev_seg, flow_x, flow_y)
            cur_seg = fixBlack(cur_seg)

            _, cur_name = adj_frame_list[i].strip('\n').split(' ')
            img_name = cur_name.split('/')[-1]
            img_name = os.path.splitext(img_name)[0]

            id_path = os.path.join(cur_seg_dir, img_name+"ID.png")
            color_path = os.path.join(cur_seg_dir, img_name+"Color.png")
            color_img = id2color(cur_seg)
            print("Writing: ", id_path)
            cv2.imwrite(id_path, cur_seg)
            print("Writing: ", color_path)
            cv2.imwrite(color_path, color_img)


if __name__ ==  "__main__":
    process()
