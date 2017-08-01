#!/usr/bin/python
from __future__ import print_function

import numpy as np
import cv2
import h5py
import argparse
import os

def process(interval):

    gt_list_file = "/media/data/bingbing/groundTruth/down-2/valGt.lst"

    root_dir = "/home/bingbing/Documents/flowProSeg"
    img_dir = os.path.join(root_dir, "interval_"+str(interval), "cur_segK2I")

    black_list_file = os.path.join(img_dir, "val_black.lst")
    part_list_file = os.path.join(img_dir, "val_part_black.lst")
    output_dir = os.path.join(img_dir, "bin_gt")


    gt_list = open(gt_list_file, 'r').readlines()
    black_list = open(black_list_file, 'r').readlines()
    part_list = open(part_list_file, 'r').readlines()


    fix_ratio = np.zeros((len(gt_list), 1))
    fix_right_ratio = np.zeros((len(gt_list), 1))

    for i in xrange(len(gt_list)):
        gt_name = gt_list[i].strip('\n')
        black_name = black_list[i].strip('\n')
        part_name = part_list[i].strip('\n')

        gt_id = cv2.imread(gt_name, 0)
        black_id = cv2.imread(black_name, 0)
        part_id = cv2.imread(part_name, 0)

        err_mask = gt_id!=part_id
        ignore_mask = gt_id==255
        err_mask[ignore_mask] = False

        black_mask = black_id==19
        part_mask = part_id==19

        fix_mask = np.copy(black_mask)
        fix_mask[part_mask] = False

        fix_ratio[i, 0] = np.sum(fix_mask)*1.0/np.sum(black_mask)

        union_mask = ignore_mask==fix_mask
        fix_mask[union_mask] = False
        fix_wrong = np.logical_and(fix_mask, err_mask)
        fix_right_ratio[i, 0] = (np.sum(fix_mask)-np.sum(fix_wrong))*1.0/np.sum(fix_mask)



    h5file_name = os.path.join(output_dir, "fix_ratio.h5")
    h5file = h5py.File(h5file_name, 'w')
    dset = h5file.create_dataset("ratio", (len(gt_list), 1), dtype='f')
    dset[...] = fix_ratio
    print(h5file_name)

    h5file_name = os.path.join(output_dir, "fix_right_ratio.h5")
    h5file = h5py.File(h5file_name, 'w')
    dset = h5file.create_dataset("ratio", (len(gt_list), 1), dtype='f')
    dset[...] = fix_right_ratio
    print(h5file_name)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, required=True, help="the interval between two frames")
    args = parser.parse_args()

    process(args.interval)

