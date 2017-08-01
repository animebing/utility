# -*- coding: utf-8 -*-
"""
Created On Sat Oct 29 21:40 2016
@Author: Jia Zheng
"""
import os
import mxnet as mx
import numpy as np


class Multi_Label_Iter(mx.io.DataIter):
    """
    Multi Label iterator.

    Args
    ----
    data_iter : DataIter
        Internal data iterator.
    """

    def __init__(self, data_iter, devs):
        super(Multi_Label_Iter, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size
        self.side = 7L
        self.ctx = mx.Context(devs[0])
        self.prob_shape    = (self.batch_size, 20L, self.side, self.side)
        self.obj_shape     = (self.batch_size, 1L, self.side, self.side)
        self.bbox_shape    = (self.batch_size, 4L, self.side, self.side)
        self.bbox_xy_shape = (self.batch_size, 2L, self.side, self.side)
        self.bbox_wh_shape = (self.batch_size, 2L, self.side, self.side)
        self.offset_shape  = (self.batch_size, 2L, self.side, self.side)

    @property
    def provide_data(self):
        """
        The name and shape of data provided by this iterator
        """
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        """
        The name and shape of label provided by this iterator
        """
        return [('prob_gt', self.prob_shape),
                ('obj_gt',  self.obj_shape),
                ('bbox_xy_gt', self.bbox_xy_shape),
                ('bbox_offset_gt', self.bbox_xy_shape),
                ('bbox_wh_gt', self.bbox_wh_shape),
                ('offset', self.offset_shape)]

    def hard_reset(self):
        """
        Ignore roll over data and set to start.
        """
        self.data_iter.hard_reset()

    def reset(self):
        """
        Reset the iterator.
        """
        self.data_iter.reset()

    def next(self):
        """
        Get next data batch from iterator.

        Returns
        -------
        data: DataBatch
            The data of next batch.
        """
        batch = self.data_iter.next()
        label = batch.label[0].asnumpy()
        # classes prob
        classes = label[:, :self.side**2].reshape(self.batch_size, self.side, self.side).astype(int)
        prob_np = np.zeros(self.prob_shape, dtype=np.float)
        idxs, rows, cols = np.where(classes)
        for idx, row, col in zip(idxs, rows, cols):
            prob_np[idx, classes[idx,row, col], row, col] = 1
        prob = mx.nd.array(prob_np, self.ctx)
        # whether the grid contains object
        obj = mx.nd.array(label[:, self.side**2:2*self.side**2], self.ctx)
        obj = obj.reshape(self.obj_shape)
        # bounding box
        bbox = label[:, 2*self.side**2:]
        bbox = bbox.reshape(self.bbox_shape)
        bbox = mx.nd.array(bbox, self.ctx)
        bbox_xy = mx.ndarray.slice_axis(src=bbox, axis=1, begin=0, end=2)
        bbox_offset = bbox_xy * self.side
        bbox_offset -= mx.ndarray.floor(bbox_offset)
        bbox_wh = mx.ndarray.slice_axis(src=bbox, axis=1, begin=2, end=4)
        # offset
        offset_x = np.repeat(np.arange(7).reshape(1,-1), 7, axis=0).reshape(1,7,7)
        offset_y = np.transpose(offset_x, (0, 2, 1))
        offset = np.concatenate((offset_x, offset_y), axis=0).reshape(1,2,7,7)
        offset = np.repeat(offset, self.batch_size, axis=0)
        offset = mx.nd.array(offset, self.ctx)
        return mx.io.DataBatch(data  = batch.data,
                               label = [prob, obj, bbox_xy, bbox_offset, bbox_wh, offset],
                               pad   = batch.pad,
                               index = batch.index)


def data_iter(batch_size, data_dir):
    data_shape = (3, 224, 224)
    train = mx.io.ImageRecordIter(
        path_imgrec  = os.path.join(data_dir, "train.rec"),
        path_imglist = os.path.join(data_dir, "train.lst"),
        data_shape   = data_shape,
        batch_size   = batch_size,
        mean_r       = 123.68,
        mean_g       = 116.779,
        mean_b       = 103.939,
        label_name   = "label",
        label_width  = 294,
        shuffle      = True,
        )
    val = mx.io.ImageRecordIter(
        path_imgrec  = os.path.join(data_dir, "val.rec"),
        path_imglist = os.path.join(data_dir, "val.lst"),
        data_shape   = data_shape,
        batch_size   = batch_size,
        mean_r       = 123.68,
        mean_g       = 116.779,
        mean_b       = 103.939,
        label_name   = "label",
        label_width  = 294,
        )
    return (train, val)
