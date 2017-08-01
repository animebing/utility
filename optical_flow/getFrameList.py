#!/usr/bin/python

from __future__ import print_function, division
import argparse


def process(data_type, interval):
    file_name = data_type+"Img.lst"
    img_list = open(file_name, 'r').readlines()

    save_file = data_type + "_interval" + '_' + str(interval) + ".lst"
    f = open(save_file, 'w')
    for i in xrange(len(img_list)):
        cur = img_list[i].strip('\n')
        tmp = cur.split('_')
        pre_frame = int(tmp[2])-interval
        pre = tmp[0]+'_'+tmp[1]+'_'+'{:0>6}'.format(pre_frame)+'_'+tmp[3]
        f.write(pre+' '+cur+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interval", type=int, required=True, help="frame interval")
    args = parser.parse_args()

    process("val", args.interval)
