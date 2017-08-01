#!/usr/bin/python
from __future__ import print_function
import os, glob

if __name__ == "__main__":


    base_dir = "/home/bingbing/Documents/flowProSeg/interval_5/cur_segK2I/val"
    pattern = os.path.join(base_dir,"*IDPartBlack.png")
    file_list = glob.glob(pattern)
    file_list.sort()

    ID_path = "interval_5/cur_segK2I/val_part_black.lst"

    fd = open(ID_path, 'w')

    for f in file_list:
        fd.write(f+'\n')






