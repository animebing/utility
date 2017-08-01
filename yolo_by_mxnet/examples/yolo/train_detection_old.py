import mxnet as mx
from data import Multi_Label_Iter, data_iter
from symbol import get_symbol
import logging
import argparse
import train_model


def get_parser():
    parser = argparse.ArgumentParser(description='Train an object detection on PASCAL VOC')
    parser.add_argument('--load-model-prefix', type=str,
                        help='the prefix of the model to load')
    parser.add_argument('--save-model-prefix', type=str,
                        help='the prefix of the model to save')
    parser.add_argument('--lr', type=float,
                        help='the initial learning rate')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    parser.add_argument('--clip-gradient', type=float,
                        help='clip min/max gradient to prevent extreme value')
    parser.add_argument('--opt', type=str, default='sgd',
                        help='the optimization method')
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--gpus', type=str, default='0',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=15680,
                        help='the number of training examples')
    parser.add_argument('--log-file', type=str,
                        help='the name of log file')
    parser.add_argument('--log-dir', type=str, default='log',
                        help='directory of the log file')
    parser.add_argument('--data-dir', type=str, default='../../dataset/',
                        help='directory of the data')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    net = get_symbol()
    arg_list = net.list_arguments()
    aux_list = net.list_auxiliary_states()
    print arg_list
    arg_shapes, output, aux_shapes = net.infer_shape(data=(1, 3, 448, 448))
    print arg_shapes

    print "---------------------------------"
    print aux_list
    print aux_shapes
    #for i in output:
     #   print i
    #train_model.fit(args, net, data_iter, Multi_Label_Iter)
