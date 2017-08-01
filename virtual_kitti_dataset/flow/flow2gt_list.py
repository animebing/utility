#!/usr/bin/python
from __future__ import print_function
import os
import sys

root_dir = "/home/bingbing/Documents/vkittiSeg/flow"
form = "p2c"
flow_dir = os.path.join(root_dir, form)
gt_dir = os.path.join(root_dir, "gt")
flow2gt_dir = os.path.join(root_dir, "flow2gt")

seqs = ["0001", "0002", "0006", "0018", "0020"]

variations = open(os.path.join(root_dir, "var.lst")).readlines()

for seq in seqs:
    for var in variations:
        var = var.strip("\n")
        flow_var_dir = os.path.join(flow_dir, seq, var)
        gt_var_dir = os.path.join(gt_dir, seq, var)
        out_txt_name = seq+"-"+var
        out_txt_file = os.path.join(flow2gt_dir, out_txt_name+".txt")
        fid = open(out_txt_file, 'w')

        flow_list = os.listdir(flow_var_dir)
        flow_list.sort()
        gt_list = os.listdir(gt_var_dir)
        gt_list.sort()

        if len(flow_list) != len(gt_list):
            sys.exit("the length of flow_list is not equal to that of gt_list")

        for i in xrange(len(flow_list)):
            flow_path = os.path.join(flow_var_dir, flow_list[i].strip("\n"))
            gt_path = os.path.join(gt_var_dir, gt_list[i].strip("\n"))

            fid.write(flow_path+" "+gt_path+"\n")

flow_seq_file = os.path.join(root_dir, "flow_seq.txt")
fid = open(flow_seq_file, 'w')

flow2gt_list = os.listdir(flow2gt_dir)
flow2gt_list.sort()
for i in xrange(len(flow2gt_list)):
    path = os.path.join(flow2gt_dir, flow2gt_list[i].strip('\n'))
    fid.write(path+"\n")

