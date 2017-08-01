#!/usr/bin/python
from __future__ import print_function
import numpy as np
import os
import sys
import cv2
import h5py
import mpi4py.MPI as MPI
import itertools
from my_util import *
from flow_process import readFlow

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


flow_seq_file = "/home/bingbing/Documents/vkittiSeg/flow/flow_seq.txt"
prop_dir = "/home/bingbing/Documents/vkittiSeg/flow/propagate/result"
gt_seg_file = "/home/bingbing/Documents/vkittiSeg/gt_seg/gt_seg.txt"
seq_num = 5
seq_list = open(flow_seq_file, 'r').readlines()
gt_seg_list = open(gt_seg_file, 'r').readlines()

if comm_rank == 0:

    choose_seq = np.random.permutation(len(seq_list))[:seq_num]
    crop_lens = [5, 10, 15]
    comb = itertools.product(choose_seq, crop_lens)
    tmp_comb = []
    for each in comb:
        tmp_comb.append(tuple(each))

    comb = tmp_comb
    offset = np.linspace(0, len(comb), comm_size+1).astype('int')
    offset_comb = [comb[offset[i]:offset[i+1]] for i in xrange(comm_size)]

else:
    offset_comb = None

local_comb = comm.scatter(offset_comb, root=0)
test_num = 10

for i, each_crop in local_comb:
    seq_one = seq_list[i].strip("\n")
    gt_seg_one = gt_seg_list[i].strip("\n")

    all_flows = open(seq_one, 'r').readlines()
    all_gt_seg = open(gt_seg_one, 'r').readlines()
    if len(all_flows) != len(all_gt_seg)-1:
        sys.exit("the length of flows is not equal to that of ground truth seg")

    seq_name = seq_one.split('/')[-1][:-4]
    seq_dir = os.path.join(prop_dir, seq_name)

    crop_dir = os.path.join(seq_dir, str(each_crop))
    random_end = np.random.randint(low=each_crop, high=len(all_flows), size=test_num)

    err_ratio = np.zeros((test_num, each_crop))
    for i in xrange(test_num):
        test_dir = os.path.join(crop_dir, str(i))
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        end = random_end[i]
        cur_seg = cv2.imread(all_gt_seg[end].strip("\n"), 0)
        pre = end
        for j in xrange(each_crop):
            pre -= 1
            # here is ground truth flow
            pre_flow_file = all_flows[pre].split(' ')[-1].strip("\n")
            output_name = pre_flow_file.split("/")[-1][:-4]
            flow_x, flow_y = readFlow(pre_flow_file)
            prop_seg = myReMap(cur_seg, flow_x, flow_y)
            cur_seg = prop_seg.copy()

            prop_color = id2color(prop_seg)

            output_seg = os.path.join(test_dir, output_name+"_pred_id.png")
            output_color = os.path.join(test_dir, output_name+"_pred_color.png")
            print("writing: ", output_seg)
            cv2.imwrite(output_seg, prop_seg)
            print("writing: ", output_color)
            cv2.imwrite(output_color, prop_color)

            gt_seg = cv2.imread(all_gt_seg[pre].strip("\n"), 0)

            err_mask = gt_seg != prop_seg
            err_mask = err_mask.astype(np.uint8)
            err_ratio[i, j] = err_mask.sum()*1.0/(err_mask.shape[0]*err_mask.shape[1])
            err_mask *= int(255)
            err_bin = os.path.join(test_dir, output_name+"_err_bin.png")
            print("writing: ", err_bin)
            cv2.imwrite(err_bin, err_mask)

    idx = list(reversed(range(each_crop)))
    err_ratio = err_ratio[:, idx]
    ratio_file = os.path.join(crop_dir, "err_ratio.h5")
    h5file = h5py.File(ratio_file, 'w')
    dset = h5file.create_dataset("ratio", (test_num, each_crop), dtype='f')
    dset[...] = err_ratio



