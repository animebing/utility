#!/usr/bin/python
from __future__ import print_function
import os
import cv2

def process(down, data_type):
    file_name = "/home/bingbing/Documents/datasets/cityscapesDataset/"+data_type+"Gt.lst"
    file_list = open(file_name, 'r').readlines()
    out_dir = os.path.join("/media/data/bingbing/groundTruth", "down-"+str(down), data_type)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    i = 0
    for f in file_list:
        label_path = f.strip('\n')
        label_name = label_path.split('/')[-1]
        label = cv2.imread(label_path, 0)
        label = cv2.resize(label, (2048/down, 1024/down), interpolation=cv2.INTER_NEAREST)

        out_path = os.path.join(out_dir, label_name)
        print("Writing label %d, %s" %(i, out_path))
        cv2.imwrite(out_path, label)
        i = i+1




if __name__ == "__main__":

    process(2, "val")
