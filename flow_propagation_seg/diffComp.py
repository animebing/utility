#!/usr/bin/python
from __future__ import print_function

import numpy as np
import cv2
import h5py
import os
import argparse
import glob

def getList(file_dir, pattern):
    path = os.path.join(file_dir, pattern)
    path_list = glob.glob(path)
    path_list.sort()
    return path_list

def process(gt_flag, interval):

    root_dir = "/home/bingbing/Documents/flowProSeg"
    if gt_flag == 1:
        bin_dir = os.path.join(root_dir, "interval_"+str(interval), "cur_segK2I", "bin_gt")
    else:
        bin_dir = os.path.join(root_dir, "interval_"+str(interval), "cur_segK2I", "bin_dila")

    diff_dir = os.path.join(bin_dir, "with_black")
    fix_dir = os.path.join(bin_dir, "without_black")

    diff_list = getList(diff_dir, "*DiffBin.png")
    fix_list = getList(fix_dir, "*ErrBin.png")
    output_dir = os.path.join(bin_dir, "bin_comp")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    iou = np.zeros((len(diff_list), 1))
    for i in xrange(len(diff_list)):

        diff_file = diff_list[i].strip('\n')
        diff_img = cv2.imread(diff_file, 0)
        diff_img /= int(255)
        diff_img = diff_img.astype(bool)

        fix_file = fix_list[i].strip('\n')
        fix_img = cv2.imread(fix_file, 0)
        fix_img /= int(255)
        fix_img = fix_img.astype(bool)

        union_img = np.logical_or(diff_img, fix_img)
        inter_img = np.logical_and(diff_img, fix_img)
        iou[i, 0] = np.sum(inter_img)*1.0/np.sum(union_img)
        union_img[inter_img] = False
        union_img = union_img.astype(np.uint8)
        union_img *= int(255)

        img_name = diff_file.split('/')[-1]
        img_name = os.path.splitext(img_name)[0]
        img_name = img_name[:-7]
        output_path = os.path.join(output_dir, img_name+"CompBin.png")
        print("Writing: ", output_path)
        cv2.imwrite(output_path, union_img)


    h5file_name = os.path.join(bin_dir, "comp_ratio.h5")
    h5file = h5py.File(h5file_name, 'w')
    dset = h5file.create_dataset('ratio', (len(diff_list), 1), dtype='f')
    dset[...] = iou



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_flag", type=int, required=True, help="1 or 0, comparison between groundtruth or dilation10")
    parser.add_argument("--interval", type=int, required=True, help="the interval between two frames")

    args = parser.parse_args()
    process(args.gt_flag, args.interval)

