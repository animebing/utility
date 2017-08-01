#!/usr/bin/python
from __future__ import print_function
import numpy as np
import cv2
import os
import sys
import itertools
import mpi4py.MPI as MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

root_dir = "/home/bingbing/Documents/vkittiSeg"
seg_gt_dir = os.path.join(root_dir, "gt_seg")
rgb_id_dir = "/home/bingbing/Documents/datasets/vkitti_1.3.1/vkitti_1.3.1_scenegt"
all_seq = ["0001", "0002", "0006", "0018", "0020"]
var_file = os.path.join(root_dir, "var.lst")
variations = open(var_file, 'r').readlines()

if comm_rank == 0:

    comb = itertools.product(all_seq, variations)
    tmp_comb = []
    for each in comb:
        tmp_comb.append(tuple(each))

    comb = tmp_comb
    offset = np.linspace(0, len(comb), comm_size+1).astype('int')
    offset_comb = [comb[offset[i]:offset[i+1]] for i in xrange(comm_size)]
else:
    offset_comb = None


local_comb = comm.scatter(offset_comb, root=0)

name2id_dict = {"Terrain":0, "Sky":1, "Tree":2, "Vegetation":3, "Building":4, "Road":5, "GuardRail":6, "TrafficSign":7, "TrafficLight":8, "Pole":9, "Misc":10, "Truck":11, "Car":12, "Van":13}

for each_comb in local_comb:
    seq, var = each_comb
    var = var.strip("\n")

    encode_txt = os.path.join(rgb_id_dir, seq+"_"+var+"_scenegt_rgb_encoding.txt")
    encode_info = open(encode_txt, 'r').readlines()
    rgb2id_dict = {}
    for each in encode_info[1:]:
        sep = each.strip("\n").split(' ')
        rgb = [int(value) for value in sep[1:] ]
        rgb = tuple(rgb)
        id_name = sep[0]
        idx = id_name.find(":")
        if idx != -1:
            id_name = id_name[:idx]

        rgb2id_dict[rgb] = name2id_dict[id_name]

    id_dir = os.path.join(seg_gt_dir, seq, var)
    if not os.path.exists(id_dir):
        os.makedirs(id_dir)

    rgb_dir = os.path.join(rgb_id_dir, seq, var)
    rgb_list = os.listdir(rgb_dir)
    rgb_list.sort()

    for each_rgb in rgb_list:
        each_rgb = each_rgb.strip("\n")
        rgb_file = os.path.join(rgb_dir, each_rgb)
        rgb_img = cv2.imread(rgb_file, 1)
        rgb_img = rgb_img[:, :, [2, 1, 0]]
        id_img_path = os.path.join(id_dir, each_rgb[:-4]+"_gt_id.png")
        height, width = rgb_img.shape[:2]
        id_img = np.zeros((height, width), dtype=np.uint8)
        for i in xrange(height):
            for j in xrange(width):
                rgb = tuple(rgb_img[i, j, :])
                if rgb not in rgb2id_dict:
                    print("the rgb value is not in rgb2id_dict")
                    id_img[i, j] = 255
                else:
                    id_img[i, j] = rgb2id_dict[rgb]
        print("writing: ", id_img_path)
        cv2.imwrite(id_img_path, id_img)


