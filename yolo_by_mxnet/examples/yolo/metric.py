# -*- coding: utf-8 -*-
"""
Created On Sat Oct 29 21:40 2016
@Author: Jia Zheng
"""
import mxnet as mx
import numpy as np
import logging


class ClassLoss(mx.metric.EvalMetric):
    """
    Evaluate classification loss.
    """
    def __init__(self):
        super(ClassLoss, self).__init__('ClassLoss')

    def update(self, labels, preds):
        """
        Update classification loss.

        Args
        ----
        labels: tuple
            (prob, obj, bbox_xy, bbox_wh, bbox_offset, offset)
        preds: tuple
            (prob, conf, bbox_offset, bbox_wh, iou)
        """
        label = preds[0].asnumpy()
        label_gt = labels[0].asnumpy()
        loss = np.square(label - label_gt).sum()
        self.sum_metric += loss
        self.num_inst += label.shape[0]


class BBoxXYLoss(mx.metric.EvalMetric):
    """
    Evaluate bounding Box xy Loss
    """
    def __init__(self):
        super(BBoxXYLoss, self).__init__('BBoxXYLoss')

    def update(self, labels, preds):
        """
        Update bounding box loss.

        Args
        ----
        labels: tuple
            (prob, obj, bbox_xy, bbox_wh, bbox_offset, offset)
        preds: tuple
            (prob, conf, bbox_offset, bbox_wh, iou)
        """
        bbox_xy = preds[2].asnumpy()
        bbox_xy_gt = labels[4].asnumpy()
        loss = np.square(bbox_xy - bbox_xy_gt).sum()
        self.sum_metric += loss
        self.num_inst += bbox_xy.shape[0]


class BBoxWHLoss(mx.metric.EvalMetric):
    """
    Evaluate bounding Box wh Loss
    """
    def __init__(self):
        super(BBoxWHLoss, self).__init__('BBoxWHLoss')

    def update(self, labels, preds):
        """
        Update bounding box loss.

        Args
        ----
        labels: tuple
            (prob, obj, bbox_xy, bbox_wh, bbox_offset, offset)
        preds: tuple
            (prob, conf, bbox_offset, bbox_wh, iou)
        """
        bbox_wh = preds[3].asnumpy()
        bbox_wh_gt = labels[3].asnumpy()
        loss = np.square(bbox_wh - bbox_wh_gt).sum()
        self.sum_metric += loss
        self.num_inst += bbox_wh.shape[0]


class ConfidenceLoss(mx.metric.EvalMetric):
    """
    Evaluate confidence loss.
    """
    def __init__(self):
        super(ConfidenceLoss, self).__init__('ConfidenceLoss')

    def update(self, labels, preds):
        """
        Update confidence loss.

        Args
        ----
        labels: tuple
            (prob, obj, bbox_xy, bbox_wh, bbox_offset, offset)
        preds: tuple
            (prob, conf, bbox_offset, bbox_wh, iou)
        """
        self.sum_metric += np.square(preds[4].asnumpy() - preds[1].asnumpy()).sum()
        self.num_inst += preds[4].shape[0]


class ClassAccuracy(mx.metric.EvalMetric):
    """
    Evaluate classification accuracy.
    """
    def __init__(self):
        super(ClassAccuracy, self).__init__('ClassAccuracy')

    def update(self, labels, preds):
        """
        Update classification accuracy.

        Args
        ----
        labels: tuple
            (prob, obj, bbox_xy, bbox_wh, bbox_offset, offset)
        preds: tuple
            (prob, conf, bbox_offset, bbox_wh, iou)
        """
        objects = labels[1].asnumpy()
        non_ignore_idxs = np.where(objects != 0)

        pred_label = mx.ndarray.argmax_channel(preds[0]).asnumpy().astype('int32')
        pred_label = pred_label.reshape((-1, 1, 7, 7))
        label = mx.ndarray.argmax_channel(labels[0]).asnumpy().astype('int32')
        label = label.reshape((-1, 1, 7, 7))

        pred_label = pred_label[non_ignore_idxs]
        label = label[non_ignore_idxs]

        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)


class AverageIOU(mx.metric.EvalMetric):
    """
    Evaluate average IOU.
    """
    def __init__(self):
        super(AverageIOU, self).__init__('AverageIOU')

    def update(self, labels, preds):
        """
        Update average IOU.

        Args
        ----
        labels: tuple
            (prob, obj, bbox_xy, bbox_wh, bbox_offset, offset)
        preds: tuple
            (prob, conf, bbox_offset, bbox_wh, iou)
        """
        self.sum_metric += np.sum(preds[4].asnumpy())
        self.num_inst += np.sum(labels[1].asnumpy())
