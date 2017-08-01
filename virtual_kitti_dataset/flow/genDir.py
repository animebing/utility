#!/usr/bin/python

from __future__ import print_function
import os
import sys

seq = sys.argv[1]

variations = open('var.lst', 'r').readlines()

for var in variations:
    var_dir = os.path.join(seq, var.strip('\n'))
    if not os.path.exists(var_dir):
        os.makedirs(var_dir)

