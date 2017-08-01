#!/usr/bin/python
from __future__ import print_function, division

import os, glob

def process(method, down, data_type, suffix):

    cwd = os.getcwd()

    pattern = os.path.join(cwd, "opticalFlow", method, "down-"+str(down), data_type, "*"+suffix+".h5")

    file_list = glob.glob(pattern)
    if not file_list:
        printError("Don't find any files")

    file_list.sort()
    list_name = "list/" + "opticalFlow" + "_" + method + "_" + "down" + "_" + str(down) + "_" + data_type + "_" + suffix + ".lst"
    fout = open(list_name, 'w')
    for f in file_list:
        fout.write(f+"\n")

if __name__ == "__main__":
    process("opencv", 4, "train", "Flow")
    process("opencv", 4, "val", "Flow")
    process("opencv", 4, "train", "FlowNorm")
    process("opencv", 4, "val", "FlowNorm")



