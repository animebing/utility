# -*- coding: utf-8 -*-
"""
Created On Sat Oct 29 21:40 2016
@Author: Jia Zheng
"""
import mxnet as mx
import math

def ConvFactory(data, num_filter, kernel, stride=(1,1)):
    pad = (kernel[0]/2, kernel[1]/2)
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    leakyRelu = mx.symbol.LeakyReLU(data=bn, slope=0.1)
    return leakyRelu



def CalculateConfidence(bbox_xy, bbox_wh, bbox_xy_gt, bbox_wh_gt):
    right = mx.symbol.minimum(bbox_xy + bbox_wh / 2.0,
                  bbox_xy_gt + bbox_wh_gt / 2.0)
    left = mx.symbol.maximum(bbox_xy - bbox_wh / 2.0,
                         bbox_xy_gt - bbox_wh_gt / 2.0)
    overlap = right - left
    overlap = mx.symbol.maximum(overlap, 0)
    w_overlap = mx.symbol.slice_axis(data=overlap, axis=1, begin=0, end=1, name='w_overlap')
    h_overlap = mx.symbol.slice_axis(data=overlap, axis=1, begin=1, end=2, name='h_overlap')
    intersection = w_overlap * h_overlap
    w_ = mx.symbol.slice_axis(data=bbox_wh, axis=1, begin=0, end=1, name='bbox_w')
    h_ = mx.symbol.slice_axis(data=bbox_wh, axis=1, begin=1, end=2, name='bbox_h')
    w_gt = mx.symbol.slice_axis(data=bbox_wh_gt, axis=1, begin=0, end=1, name='bbox_w_gt')
    h_gt = mx.symbol.slice_axis(data=bbox_wh_gt, axis=1, begin=1, end=2, name='bbox_h_gt')
    union = w_ * h_ + w_gt * h_gt - intersection
    iou = intersection / union
    return iou


def get_symbol():
    num_grids = 7
    num_classes = 20
    num_out = 7*7*(4+1+20)
    data = mx.symbol.Variable(name="data")
    prob_gt    = mx.symbol.Variable(name="prob_gt")  # format, assuming 7*7*20 dimension, order: TBD
    obj_gt     = mx.symbol.Variable(name="obj_gt")  # format, assuming 0 or 1, 7*7, order: TBD
    bbox_xy_gt = mx.symbol.Variable(name="bbox_xy_gt") # format, assuming 7*7*4, order: TBD
    bbox_offset_gt = mx.symbol.Variable(name="bbox_offset_gt")
    bbox_wh_gt = mx.symbol.Variable(name="bbox_wh_gt")
    offset     = mx.symbol.Variable(name="offset")
    # conv stage 1
    conv_1 = ConvFactory(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2))
    pool_1 = mx.symbol.Pooling(data=conv_1, kernel=(2, 2), pool_type='max', stride=(2, 2))

    # conv stage 2
    conv_2 = ConvFactory(data=pool_1, num_filter=192, kernel=(3, 3), stride=(1, 1))
    pool_2 = mx.symbol.Pooling(data=conv_2, kernel=(2, 2), pool_type='max', stride=(2, 2))

    #s conv stage 3
    conv_3 = ConvFactory(data=pool_2, num_filter=128, kernel=(1, 1), stride=(1, 1))
    conv_3 = ConvFactory(data=conv_3, num_filter=256, kernel=(3, 3), stride=(1, 1))
    conv_3 = ConvFactory(data=conv_3, num_filter=256, kernel=(1, 1), stride=(1, 1))
    conv_3 = ConvFactory(data=conv_3, num_filter=512, kernel=(3, 3), stride=(1, 1))
    pool_3 = mx.symbol.Pooling(data=conv_3, kernel=(2, 2), pool_type='max', stride=(2, 2))

    # conv stage 4
    conv_4 = ConvFactory(data=pool_3, num_filter=256, kernel=(1, 1), stride=(1, 1))
    conv_4 = ConvFactory(data=conv_4, num_filter=512, kernel=(3, 3), stride=(1, 1))
    for i in range(3):
        conv_4 = ConvFactory(data=conv_4, num_filter=256, kernel=(1, 1), stride=(1, 1))
        conv_4 = ConvFactory(data=conv_4, num_filter=512, kernel=(3, 3), stride=(1, 1))
    conv_4 = ConvFactory(data=conv_4, num_filter=512, kernel=(1, 1), stride=(1, 1))
    conv_4 = ConvFactory(data=conv_4, num_filter=1024, kernel=(3, 3), stride=(1, 1))
    pool_4 = mx.symbol.Pooling(data=conv_4, kernel=(2, 2), pool_type='max', stride=(2, 2))

    # conv stage 5
    conv_5 = ConvFactory(data=pool_4, num_filter=512, kernel=(1, 1), stride=(1, 1))
    conv_5 = ConvFactory(data=conv_5, num_filter=1024, kernel=(3, 3), stride=(1, 1))

    conv_5 = ConvFactory(data=conv_5, num_filter=512, kernel=(1, 1), stride=(1, 1))
    conv_5 = ConvFactory(data=conv_5, num_filter=1024, kernel=(3, 3), stride=(1, 1))

    conv_5 = ConvFactory(data=conv_5, num_filter=1024, kernel=(3, 3), stride=(1, 1))
    conv_5 = ConvFactory(data=conv_5, num_filter=1024, kernel=(3, 3), stride=(2, 2))

    # conv stage 6
    conv_6 = ConvFactory(data=conv_5, num_filter=1024, kernel=(3, 3), stride=(1, 1))
    conv_6 = ConvFactory(data=conv_6, num_filter=1024, kernel=(3, 3), stride=(1, 1))
    return conv_6
    """
    # conn stage 7
    fc_7 = mx.symbol.FullyConnected(data=conv_6, num_hidden=4096)
    drop_7 = mx.symbol.Dropout(data=fc_7, p=0.5)
    output = mx.symbol.FullyConnected(data=drop_7, num_hidden=num_out)
    output = mx.symbol.Reshape(data=output, shape=(-1, 25, 7, 7))
    out_conf = mx.symbol.slice_axis(data=output, axis=1, begin=0, end=1)   # [1, 7, 7]
    out_conf = mx.symbol.Activation(data=out_conf, act_type='sigmoid')

    out_bbox = mx.symbol.slice_axis(data=output, axis=1, begin=1, end=5)   # [4, 7, 7]
    out_bbox = mx.symbol.Activation(data=out_bbox, act_type='sigmoid')

    out_classes = mx.symbol.slice_axis(data=output, axis=1, begin=5, end=25)   # [20, 7, 7]


    # Classes loss
    prob_softmax = mx.symbol.SoftmaxActivation(data=out_classes, mode='channel')
    prob_obj = mx.symbol.broadcast_mul(prob_softmax, obj_gt, name='prob_obj')
    prob_loss = mx.symbol.LinearRegressionOutput(data=prob_obj,
                                                 label=prob_gt,
                                                 name='prob_loss')

    # bounding box loss

    bbox_offset = mx.symbol.slice_axis(data=out_bbox, axis=1, begin=0, end=2, name='bbox_offset')
    bbox_wh = mx.symbol.slice_axis(data=out_bbox, axis=1, begin=2, end=4, name='bbox_wh')

    bbox_wh_sqrt = mx.symbol.sqrt(bbox_wh, name='bbox_wh_sqrt')
    bbox_wh_gt_sqrt = mx.symbol.sqrt(bbox_wh_gt, name='bbox_wh_gt_sqrt')
    bbox_wh_gt_no_grad = mx.symbol.BlockGrad(data=bbox_wh_gt_sqrt, name='bbox_wh_gt_no_grad')
    # only penalize error if an object is present in the grid cell
    bbox_offset_obj = mx.symbol.broadcast_mul(bbox_offset, obj_gt, name='bbox_offset_obj')
    bbox_wh_obj = mx.symbol.broadcast_mul(bbox_wh_sqrt, obj_gt, name='bbox_wh_obj')
    bbox_offset_loss = mx.symbol.LinearRegressionOutput(data=bbox_offset_obj,
                                                    label=bbox_offset_gt,
                                                    grad_scale=5,
                                                    name='bbox_xy_loss')
    bbox_wh_loss = mx.symbol.LinearRegressionOutput(data=bbox_wh_obj,
                                                    label=bbox_wh_gt_no_grad,
                                                    grad_scale=5,
                                                    name='bbox_wh_loss')


    # confidence loss
    weight_mask = 0.5*obj_gt + 0.5
    weight_mask = mx.symbol.sqrt(weight_mask)
    conf_act_weight = mx.symbol.broadcast_mul(out_conf, weight_mask)

    bbox_xy = mx.symbol.broadcast_mul(obj_gt, (bbox_offset + offset) / 7.0)
    conf_gt = CalculateConfidence(bbox_xy, bbox_wh, bbox_xy_gt, bbox_wh_gt)
    conf_gt_weight = mx.symbol.broadcast_mul(conf_gt, weight_mask)
    conf_gt_no_grad = mx.symbol.BlockGrad(data=conf_gt_weight, name='conf_gt_no_grad')
    conf_loss = mx.symbol.LinearRegressionOutput(data=conf_act_weight,
                                                 label=conf_gt_no_grad,
                                                 name='conf_loss')

    conf_gt_output = mx.symbol.MakeLoss(conf_gt_no_grad, name='conf_gt_label')
    # group error
    group = mx.symbol.Group([prob_loss, conf_loss, bbox_offset_loss, bbox_wh_loss, conf_gt_output])

    #group = mx.symbol.Group([conv_1, pool_1, conv_2, pool_2, conv_3, pool_3, conv_4, pool_4, conv_5, conv_6, fc_7, output])
    return group
    """
