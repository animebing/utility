# -*- coding: utf-8 -*-
"""
Created On Sun Oct 9 17:15 2016
@Author: Jia Zheng
"""

import os
import random
from itertools import repeat

# sets = [('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
train_sets = [('2012', 'train'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
test_sets = [('2012', 'val')]

CLASSES = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', \
            'bus', 'car', 'cat', 'chair', 'cow', \
            'diningtable', 'dog', 'horse', 'motorbike', 'person', \
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

IMAGEID_FORMAT = 'VOCdevkit/VOC{0}/ImageSets/Main/{1}.txt'
LABEL_FORMAT = 'VOCdevkit/VOC{0}/Labels/{1}.txt'
DATA_FORMAT = 'VOCdevkit/VOC{0}/JPEGImages/{1}.jpg'
LINE_FORMAT = '{0}\t{1}\t{2}\n'


def get_label(lines):
  """
  Convert label by using.

  Args
  -------
  lines: str
    line format: [class_id, x, y, w, h]

  Returns
  -------
  label: str
    str format:
    [ class[1-20]:                side^2,
      object in grid[bool]:       side^2,
      BBox(x, y, w, h)[flost]:  4*side^2]
  """
  side = 7
  label = list(repeat('0', side*side*(4+1+1)))
  for line in lines:
    line = line.strip().split()
    col = int(float(line[1]) * side)
    row = int(float(line[2]) * side)
    # class
    label[row*side+col] = line[0]
    # object
    label[row*side+col + side**2] = str(1)
    # bbox
    idx = 2 * side**2 + row * side + col # index begin
    label[idx::side**2] = line[1:]
  return "\t".join(label)

def make_list(year, image_set, list_name):
  """
  Make training or validation list

  Args
  ------
  year: str
  image_set: str
  list_name: str
  """
  list_file = open(list_name, 'a')
  image_ids = open(IMAGEID_FORMAT.format(year, image_set)).read().strip().split()
  index = 0
  for image_id in image_ids:
    try:
      lines = open(LABEL_FORMAT.format(year, image_id)).readlines()
    except:
      print 'no file VOCdevkit/VOC%s/Labels/%s.txt'%(year, image_id)
      continue
    index += 1
    label = get_label(lines)
    list_file.write(LINE_FORMAT.format(index, label, DATA_FORMAT.format(year, image_id)))
  list_file.close()
  print "VOC%s%s Total: %d images" % (year, image_set, index)

def shuffle_list(list_name):
  """
  Shuffle list

  Args
  ---------
  list_name: str
  """
  lines = open(list_name, 'r').readlines()
  # shuffle list
  random.shuffle(lines)
  list_file = open(list_name, 'w')
  for line in lines:
    list_file.write(line)
  list_file.close()

# train.lst
list_name = 'train.lst'
if os.path.exists(list_name):
  os.remove(list_name)
print 'make %s'%(list_name, )
for year, image_set in train_sets:
  make_list(year, image_set, list_name)
shuffle_list(list_name)

# val.lst
list_name = 'val.lst'
if os.path.exists(list_name):
  os.remove(list_name)
print 'make %s'%(list_name, )
for year, image_set in test_sets:
  make_list(year, image_set, list_name)
shuffle_list(list_name)
