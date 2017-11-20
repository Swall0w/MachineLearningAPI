import argparse
import chainer
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv import utils
import os
import socket

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default='voc0712')
    parser.add_argument('src_dir')
    parser.add_argument('dst_dir')
    return  parser.parse_args()


def main():
    args = arg()
    chainer.config.train = False

    model = SSD300(
        n_fg_class=len(voc_bbox_label_names),
        pretrained_model=args.pretrained_model
    )

    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        chainer.cuda.get_device(args.gpu).use()


if __name__ == '__main__':
    main()
