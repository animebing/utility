#!/usr/bin/python
from __future__ import print_function, division

import os

def process(method, down, data_type):

    cwd = os.getcwd()
    file_dir = os.path.join(cwd, "seg19", method,"down-"+str(down), data_type)
    file_list = os.listdir(file_dir)
    file_list.sort()

    list_name = "list/" + "seg19" + '_' + method + '_' + "down_" + str(down) + '_' + data_type + ".lst"
    fd = open(list_name, 'w')
    for f in file_list:
        fd.write(file_dir+'/'+f+'\n')


if __name__ == "__main__":

    process("dilation10", 8, "train")
    process("dilation10", 8, "val")
