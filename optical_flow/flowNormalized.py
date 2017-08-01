#!/usr/bin/python

from __future__ import print_function
import os
import numpy as np
import h5py

def process(method, down, data_type):

    cwd = os.getcwd()
    file_dir = os.path.join("opticalFlow", method, "down-"+str(down), data_type)
    file_list = os.listdir(file_dir)
    file_list.sort()

    for f in file_list:
        in_file = os.path.join(file_dir, f)
        fin = h5py.File(in_file, 'r')
        data = np.array(fin['optical_flow'])
        data = data[0]
        for i in xrange(data.shape[0]):
            min_v = np.min(data[i])
            max_v = np.max(data[i])
            data[i] = (data[i]-min_v)/(max_v-min_v)

        data = data[np.newaxis, :, :, :]

        h_name = f.split('.')[0]
        out_name = h_name + "Norm.h5"
        out_file = os.path.join(file_dir, out_name)
        fout = h5py.File(out_file, 'w')
        dset = fout.create_dataset("optical_flow", data.shape, dtype='f')
        dset[...] = data
        print("writing", out_file)
        #break;


if __name__ == "__main__":

    process("opencv", 4, "train")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    process("opencv", 4, "val")
