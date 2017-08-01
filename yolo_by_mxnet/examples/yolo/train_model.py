import mxnet as mx
import numpy as np
from callback import draw_bbox
from metric import *
import logging
import os


def fit(args, network, data_loader, Multi_Label_Iter, batch_end_callback=None):
    # logging
    head = '%(asctime)-15s %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    load_model_prefix = args.load_model_prefix
    model_args = {}
    if args.load_epoch is not None:
        assert load_model_prefix is not None
        tmp = mx.model.FeedForward.load(load_model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.load_epoch}
    # save model
    save_model_prefix = args.save_model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # data
    (train, val) = data_loader(args.batch_size, args.data_dir)
    train = Multi_Label_Iter(train, devs)
    val   = Multi_Label_Iter(val, devs)

    epoch_size = args.num_examples / args.batch_size
    model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
        step = max(int(epoch_size * args.lr_factor_epoch), 1),
        factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    if args.opt == 'sgd':
        model_args['optimizer'] = 'sgd'
        model_args['momentum']  = 0.9
        model_args['wd']        = 0.0005
    elif args.opt == 'rmsprop':
        model_args['optimizer'] = 'rmsprop'
    else:
        logger.info("Wrong optimizer: %s", args.opt)
        raise

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args)

    eval_metrics = [ClassLoss(),
                    BBoxXYLoss(),
                    BBoxWHLoss(),
                    ConfidenceLoss(),
                    ClassAccuracy(),
                    AverageIOU()]

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 10))

    epoch_end_callback = [
                          #draw_bbox(val, devs),
                          checkpoint
                         ]

    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = eval_metrics,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = checkpoint)
