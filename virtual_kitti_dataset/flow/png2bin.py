#!/usr/bin/python
from __future__ import print_function
import numpy as np
import cv2
import h5py
import sys
import struct
import os


def png_flow_bin(img_name, in_dir, out_dir):
    "Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"
    # read png to bgr in 16 bit unsigned short
    flow_fn = os.path.join(in_dir, img_name)
    bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag == 0 for sky or other invalid flow
    invalid = bgr[..., 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 - 1]
    out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
    out_flow[..., 0] *= w - 1
    out_flow[..., 1] *= h - 1
    out_flow[invalid] = 0  # or another value (e.g., np.nan)

    flow = out_flow.transpose(2, 0, 1)

    _, h, w = flow.shape
    out_file = os.path.join(out_dir, img_name[:-4]+".flo")
    fid = open(out_file, "wb")
    fid.write("PIEH")
    tmp = struct.pack("ii", w, h)
    fid.write(tmp)
    for i in xrange(h):
        for j in xrange(w):
            tmp = struct.pack("ff", flow[0, i, j], flow[1, i, j])
            fid.write(tmp)
    print(out_file)


in_dir = "/home/bingbing/Documents/datasets/vkitti_1.3.1/vkitti_1.3.1_flowgt"
out_dir = "/home/bingbing/Documents/vkittiSeg/flow/gt"

seq = sys.argv[1]

variations = open("var.lst", 'r').readlines()

for var in variations:
    var = var.strip("\n")
    in_var_dir = os.path.join(in_dir, seq, var)
    out_var_dir = os.path.join(out_dir, seq, var)
    if not os.path.exists(out_var_dir):
        os.makedirs(out_var_dir)

    img_files = os.listdir(in_var_dir)
    img_files.sort()
    for i in xrange(len(img_files)):
        png_flow_bin(img_files[i].strip("\n"), in_var_dir, out_var_dir)



