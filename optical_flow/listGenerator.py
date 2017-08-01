#!/usr/bin/python
from __future__ import print_function
import os, glob

if __name__ == "__main__":

    """
    base_dir = "/media/data/bingbing/seg19"
    pattern = os.path.join(base_dir, "dilation10", "down-1", "train", "*ID.png")
    file_list = glob.glob(pattern)
    file_list.sort()

    ID_path = os.path.join(base_dir, "dilation10", "down-4", "val", "ID.lst")

    fd = open(ID_path, 'w')

    for f in file_list:
        fd.write(f+'\n')
    """

    base_dir = "/media/data/bingbing/groundTruth/down-2"
    pattern = os.path.join(base_dir, "val", "*.png")
    file_list = glob.glob(pattern)
    file_list.sort()

    ID_path = os.path.join(base_dir, "valGt.lst")

    fd = open(ID_path, 'w')

    for f in file_list:
        fd.write(f+'\n')


    """
    base_dir = "/media/data/bingbing/groundTruth"
    pattern = os.path.join(base_dir, "down-4", "val", "*Ids.png")
    file_list = glob.glob(pattern)
    file_list.sort()

    ID_path = os.path.join(base_dir, "down-4", "val", "ID.lst")

    fd = open(ID_path, 'w')

    for f in file_list:
        fd.write(f+'\n')
    """




