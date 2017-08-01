#!/usr/bin/python
from __future__ import print_function

import numpy as np
import cv2
import h5py
import argparse
import os

def process(ratio, interval):

    gt_list_file = "/media/data/bingbing/groundTruth/down-2/valGt.lst"
    train_val = "val"
    root_dir = "/home/bingbing/Documents/flowProSeg"
    img_dir = os.path.join(root_dir, "interval_"+str(interval), "cur_segK2I")
    if ratio == 1:
        img_list_file = os.path.join(img_dir, "val_part_black.lst")
        output_dir = os.path.join(img_dir, "bin_gt", "with_part_black")
    else:
        img_list_file = os.path.join(img_dir, "val_out.lst")
        output_dir = os.path.join(img_dir, "bin_gt", "without_black")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    gt_list = open(gt_list_file, 'r').readlines()
    img_list = open(img_list_file, 'r').readlines()


    err_ratio = np.zeros((len(gt_list), 1))
    for i in xrange(len(gt_list)):
        gt_name = gt_list[i].strip('\n')
        img_name = img_list[i].strip('\n')
        gt_id = cv2.imread(gt_name, 0)
        img_id = cv2.imread(img_name, 0)
        print(img_name)

        err_mask = gt_id!=img_id
        ignore_mask = gt_id==255
        err_mask[ignore_mask] = False

        if ratio == 1:
            mask_19 = img_id==19
            union_mask = ignore_mask==mask_19
            mask_19[union_mask] = False
            err_ratio[i, 0] = np.sum(mask_19)*1.0/np.sum(err_mask)

            mask_diff = err_mask!=mask_19

        err_mask = err_mask.astype(np.uint8)
        err_mask *= int(255)
        output_name = img_name.split('/')[-1]
        output_name = os.path.splitext(output_name)[0]
        if ratio == 1:
            output_name = output_name[:-5]

        output_path = os.path.join(output_dir, output_name+"ErrBin.png")
        print("Writing: ", output_path)
        cv2.imwrite(output_path, err_mask)
        if ratio == 1:
            output_path = os.path.join(output_dir, output_name+"BlackBin.png")
            mask_19 = mask_19.astype(np.uint8)
            mask_19 *= int(255)
            print("Writing: ", output_path)
            cv2.imwrite(output_path, mask_19)

            output_path = os.path.join(output_dir, output_name+"DiffBin.png")
            mask_diff = mask_diff.astype(np.uint8)
            mask_diff *= int(255)
            print("Writing: ", output_path)
            cv2.imwrite(output_path, mask_diff)


    if ratio == 1:
        h5file_name = os.path.join(output_dir, "black_ratio.h5")
        h5file = h5py.File(h5file_name, 'w')
        dset = h5file.create_dataset("bin", (len(gt_list), 1), dtype='f')
        dset[...] = err_ratio


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ratio", type=int, default=1, help="1 or 0, whether to compute the ratio of black area")
    parser.add_argument("--interval", type=int, required=True, help="the interval between two frames")
    args = parser.parse_args()

    process(args.ratio, args.interval)

