#!/usr/bin/python
from __future__ import print_function
import os
import sys

root_dir = "/home/bingbing/Documents/vkittiSeg"
gt_dir = os.path.join(root_dir, "gt_seg")
gt_list_dir = os.path.join(gt_dir, "gt_list")

seqs = ["0001", "0002", "0006", "0018", "0020"]

variations = open(os.path.join(root_dir, "var.lst")).readlines()

for seq in seqs:
    for var in variations:
        var = var.strip("\n")
        gt_var_dir = os.path.join(gt_dir, seq, var)

        out_txt_name = seq+"-"+var
        out_txt_file = os.path.join(gt_list_dir, out_txt_name+".txt")
        fid = open(out_txt_file, 'w')

        img_list = os.listdir(gt_var_dir)
        img_list.sort()


        for i in xrange(len(img_list)):
            id_path = os.path.join(gt_var_dir, img_list[i].strip("\n"))

            fid.write(id_path+"\n")

gt_seg_file = os.path.join(gt_dir, "gt_seg.txt")
fid = open(gt_seg_file, 'w')

all_list = os.listdir(gt_list_dir)
all_list.sort()
for i in xrange(len(all_list)):
    path = os.path.join(gt_list_dir, all_list[i].strip('\n'))
    fid.write(path+"\n")

