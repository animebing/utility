#!/usr/bin/python
from __future__ import print_function
import struct
import numpy as np
import h5py
import sys


def readFlow(flow_path):
    flow = open(flow_path, 'r')

    flow.seek(4)
    width = struct.unpack("i", flow.read(4))[0]
    height = struct.unpack("i", flow.read(4))[0]

    flow_x = np.zeros((height, width), dtype="float")
    flow_y = np.zeros((height, width), dtype="float")

    for i in xrange(height):
	for j in xrange(width):
	    tmp = struct.unpack("f", flow.read(4))
	    flow_x[i, j] = tmp[0]
	    tmp = struct.unpack("f", flow.read(4))
	    flow_y[i, j] = tmp[0]
    return (flow_x, flow_y)
