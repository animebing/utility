# -*- coding: utf-8 -*-
"""
Created On Sat Nov 05 12:55 2016
@Author: Jia Zheng
"""
import cv2
import mxnet as mx
import numpy as np
import logging


classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def draw_bbox(val_data, ctx):
    """
    Callback to draw bounding box every epoch.

    Args
    ----
    val_data : DataIter
        Validation Data Iterator.
    ctx : Context
        The device context

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_epoch_callback to fit.
    """
    def draw_bboxs(image, objects, labels, gt):
        """
        Draw bounding box

        Args
        ----
        image: numpy.array
            image
        objects: mx.nd.array
            whether the grid contains an object
        labels: tuple
            if gt is True:
                (prob, obj, bbox_xy, bbox_wh, bbox_offset, offset)
            else:
                (prob, conf, bbox_offset, bbox_wh, iou)
        gt: bool
            BBox is ground truth or not
        """
        (rows, cols) = np.where(objects.asnumpy()[0][0])
        if gt == True:
            bbox_xy = labels[2].asnumpy()[0]
        else:
            bbox_offset = labels[2].asnumpy()[0]
            offset_x = np.repeat(np.array(range(7)).reshape(1,-1), 7, axis=0).reshape(1,7,7)
            offset_y = np.transpose(offset_x, (0, 2, 1))
            offset = np.concatenate((offset_x, offset_y), axis=0)
            bbox_xy = objects.asnumpy()[0][0] * (bbox_offset + offset) / 7.0
        bbox_wh = labels[3].asnumpy()[0]

        class_gt = mx.ndarray.argmax_channel(labels[0]).asnumpy()[0].astype('int32').reshape(7,7)
        bbox_xy = bbox_xy * 224
        bbox_wh = bbox_wh * 224
        for (row, col) in zip(rows, cols):
            xmin = int(max(0, np.around(bbox_xy[(0, row, col)] - bbox_wh[(0, row, col)] / 2.0)))
            ymin = int(max(0, np.around(bbox_xy[(1, row, col)] - bbox_wh[(1, row, col)] / 2.0)))
            xmax = int(min(223, np.around(bbox_xy[(0, row, col)] + bbox_wh[(0, row, col)] / 2.0)))
            ymax = int(min(223, np.around(bbox_xy[(1, row, col)] + bbox_wh[(1, row, col)] / 2.0)))
            #logging.info('%s \t xmin:%s \t ymin:%s \t xmax:%s \t ymax:%s' %
            #        (classes[class_gt[(row, col)]], xmin, ymin, xmax, ymax))
            image[ymin, xmin:xmax+1, 0] = 255 if gt else 0
            image[ymin, xmin:xmax+1, 1] = 0 if gt else 255
            image[ymin, xmin:xmax+1, 2] = 0

            image[ymin:ymax+1, xmin, 0] = 255 if gt else 0
            image[ymin:ymax+1, xmin, 1] = 0 if gt else 255
            image[ymin:ymax+1, xmin, 2] = 0

            image[ymax, xmin:xmax+1, 0] = 255 if gt else 0
            image[ymax, xmin:xmax+1, 1] = 0 if gt else 255
            image[ymax, xmin:xmax+1, 2] = 0

            image[ymin:ymax+1, xmax, 0] = 255 if gt else 0
            image[ymin:ymax+1, xmax, 1] = 0 if gt else 255
            image[ymin:ymax+1, xmax, 2] = 0
        return image


    def _callback(iter_no, sym, arg, aux):
        """
        The draw bounding box function.
        """
        data_names = val_data.provide_data
        label_names = val_data.provide_label
        val_data.reset()
        batch = val_data.next()
        data = batch.data
        labels = batch.label
        arg[data_names[0][0]] = data[0]
        for (label_name, label) in zip(label_names, labels):
            arg[label_name[0]] = label
        # input image
        image = data[0].asnumpy()[0]
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 0, 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image += np.array([123.68, 116.779, 103.939]).reshape(1,1,3).astype(np.uint8)
        # output
        exector = sym.bind(ctx = ctx[0], args = arg, aux_states = aux)
        exector.forward()
        # bbox predict
        image = draw_bboxs(image, labels[1], exector.outputs, False)
        # bbox ground truth
        image = draw_bboxs(image, labels[1], labels, True)
        cv2.imwrite('bbox/bbox_%03d.jpg' % iter_no, image)
    return _callback
