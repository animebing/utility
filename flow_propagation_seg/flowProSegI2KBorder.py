#!/usr/bin/python

from __future__ import print_function
import os
import cv2
import numpy as np
import h5py
import json


info_path = "/home/bingbing/git/dilation/datasets/cityscapes.json"
with open(info_path, 'r') as fp:
    info = json.load(fp)
palette = np.array(info["palette"], dtype=np.uint8)
black = np.array([[0, 0, 0]], dtype=np.uint8)
palette = np.concatenate([palette, black], axis=0)


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


def process():

    root_dir = "/home/bingbing/Documents/flowProSeg/interval_5"
    train_val = "val"
    flow_dir = os.path.join(root_dir, "spynet_flowI2K", train_val)

    prev_seg_dir = os.path.join(root_dir, "prev_seg", train_val)
    cur_seg_dir = os.path.join(root_dir, "cur_segI2K", train_val)
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
