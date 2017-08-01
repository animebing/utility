# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pickle
import os
from os.path import join

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", \
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    """
    Normalize bbox

    Args
    -------
    size: tuple
        Image size = [width, height]
    box: tuple
        bbox coordinate = [xmin, xmax, ymin, ymax]

    Returns
    -------
    Normalized bbox coordinate (x, y, w, h)
    """
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    """
    Extract labels from annotations xml file
    and save labels in a txt file.

    Args
    -------
    year: str
        year in sets
    image_id: str
        image_id in sets
    """
    try:
        in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    except:
        print 'no file VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id)
        return
    out_file = open('VOCdevkit/VOC%s/Labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/Labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/Labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    print 'Read VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)
    for image_id in image_ids:
        convert_annotation(year, image_id)
