# -*- coding: utf-8 -*-
"""
Created On Sat Oct 29 21:40 2016
@Author: Jia Zheng
"""
import mxnet as mx
import numpy as np
import logging


def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    bn = mx.symbol.BatchNorm(data=conv, name='bn_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=bn, act_type='relu', name='relu_%s%s' %(name, suffix))
    return act

def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def InceptionFactoryB(data, num_3x3red, num_3x3, num_d3x3red, num_d3x3, name):
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1),  name=('%s_double_3x3' % name), suffix='_reduce')
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_double_3x3_0' % name))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_double_3x3_1' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max", name=('max_pool_%s_pool' % name))
    # concat
    concat = mx.symbol.Concat(*[c3x3, cd3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat


def InceptionBN(data):
    """
    pretrained model
    """
    # stage 1
    conv1 = ConvFactory(data=data, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), name='conv1')
    pool1 = mx.symbol.Pooling(data=conv1, kernel=(3, 3), stride=(2, 2), name='pool1', pool_type='max')
    # stage 2
    conv2red = ConvFactory(data=pool1, num_filter=64, kernel=(1, 1), stride=(1, 1), name='conv2red')
    conv2 = ConvFactory(data=conv2red, num_filter=192, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='conv2')
    pool2 = mx.symbol.Pooling(data=conv2, kernel=(3, 3), stride=(2, 2), name='pool2', pool_type='max')
    # stage 2
    in3a = InceptionFactoryA(pool2, 64, 64, 64, 64, 96, "avg", 32, '3a')
    in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, "avg", 64, '3b')
    in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, '3c')
    # stage 3
    in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, "avg", 128, '4a')
    in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128, "avg", 128, '4b')
    in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, "avg", 128, '4c')
    in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192, "avg", 128, '4d')
    in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, '4e')
    # stage 4
    in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, "avg", 128, '5a')
    in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, "max", 128, '5b')
    return in5b

def CalculateConfidence(bbox_xy, bbox_wh, bbox_xy_gt, bbox_wh_gt):
    # Calculate IOU
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

def get_symbol(num_classes=20):
    # data & label
    data       = mx.symbol.Variable(name="data")
    prob_gt    = mx.symbol.Variable(name="prob_gt")
    obj_gt     = mx.symbol.Variable(name="obj_gt")
    bbox_xy_gt = mx.symbol.Variable(name="bbox_xy_gt")
    bbox_offset_gt = mx.symbol.Variable(name="bbox_offset_gt")
    bbox_wh_gt = mx.symbol.Variable(name="bbox_wh_gt")
    offset     = mx.symbol.Variable(name="offset")

    # pretrained model
    inception = InceptionBN(data) # [1024, 7, 7]

    # Class Probability [20, 7, 7]
    prob_conv = mx.symbol.Convolution(data=inception, num_filter=num_classes, name='prob_conv',
                                      kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    prob_softmax = mx.symbol.SoftmaxActivation(data=prob_conv, mode='channel', name='prob_softmax')
    prob_obj = mx.symbol.broadcast_mul(prob_softmax, obj_gt, name='prob_obj')
    prob_loss = mx.symbol.LinearRegressionOutput(data=prob_obj,
                                                 label=prob_gt,
                                                 name='prob_loss')

    # Bounding Box [4, 7, 7]
    bbox_conv = mx.symbol.Convolution(data=inception, num_filter=4, name='bbox_conv',
                                      kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    bbox_act = mx.symbol.Activation(data=bbox_conv, act_type='sigmoid', name='bbox_act')
    # slice bbox to bbox_offset and bbox_wh
    bbox_offset = mx.symbol.slice_axis(data=bbox_act, axis=1, begin=0, end=2, name='bbox_offset')
    bbox_wh = mx.symbol.slice_axis(data=bbox_act, axis=1, begin=2, end=4, name='bbox_wh')
    bbox_wh_sqrt = mx.symbol.sqrt(bbox_wh, name='bbox_wh_sqrt')
    bbox_wh_gt_sqrt = mx.symbol.sqrt(bbox_wh_gt, name='bbox_wh_gt_sqrt')
    bbox_wh_gt_no_grad = mx.symbol.BlockGrad(data=bbox_wh_gt_sqrt, name='bbox_wh_gt_no_grad')
    # only penalize error if an object is present in the grid cell
    bbox_offset_obj = mx.symbol.broadcast_mul(bbox_offset, obj_gt, name='bbox_offset_obj')
    bbox_wh_obj = mx.symbol.broadcast_mul(bbox_wh_sqrt, obj_gt, name='bbox_wh_obj')
    bbox_offset_loss = mx.symbol.LinearRegressionOutput(data=bbox_offset_obj,
                                                    label=bbox_offset_gt,
                                                    grad_scale=5,
                                                    name='bbox_offset_loss')
    bbox_wh_loss = mx.symbol.LinearRegressionOutput(data=bbox_wh_obj,
                                                    label=bbox_wh_gt_no_grad,
                                                    grad_scale=5,
                                                    name='bbox_wh_loss')

    # Confidence [1, 7, 7]
    weight_mask = 0.5 * obj_gt + 0.5
    weight_mask = mx.symbol.sqrt(weight_mask, name='weight_mask')
    conf_conv = mx.symbol.Convolution(data=inception, num_filter=1, name='conf_conv',
                                      kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    conf_act = mx.symbol.Activation(data=conf_conv, act_type='sigmoid', name='conf_act')
    conf_act_weight = mx.symbol.broadcast_mul(conf_act, weight_mask)
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
    return group
