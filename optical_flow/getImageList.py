#!/usr/bin/python
from __future__ import print_function, division

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_type", type=str, default="seg",
                        help="seg or opticalFlow")
    parser.add_argument("--method", type=str, default="opencv",
                        help="method to compute seg or opticalFlow")
    parser.add_argument("--down", type=int, default=2,
                        help="multipl to downsample")
    parser.add_argument("--data_type", type=str, default="train",
                        help="train or val data")

    args = parser.parse_args()

    cwd = os.getcwd()
    file_dir = os.path.join(cwd, args.image_type, args.method,"down-"+str(args.down), args.data_type)
    file_list = os.listdir(file_dir)
    file_list.sort()

    list_name = "list/"+args.image_type+'_'+args.method+'_'+"down_"+str(args.down)+'_'+args.data_type+".lst"
    fd = open(list_name, 'w')
    i = 0;
    for f in file_list:
        fd.write(file_dir+'/'+f+' '+str(i)+'\n')
        i = i+1

