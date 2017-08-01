#!/usr/bin/python
from __future__ import print_function, division
import os


def process(down, data_type):
    cwd = os.getcwd()
    file_dir = os.path.join(cwd, "groundTruth", "down-"+str(down), data_type)
    file_list = os.listdir(file_dir)
    file_list.sort()

    list_name = "list/"+"gt"+'_'+"down_"+str(down)+'_'+data_type+".lst"
    fd = open(list_name, 'w')
    i = 0;
    for f in file_list:
        fd.write(file_dir+'/'+f+' '+str(i)+'\n')
        i = i+1

if __name__ == "__main__":

    process(4, "train")
    process(4, "val")
