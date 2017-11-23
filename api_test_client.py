import argparse
import chainer
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv import utils
import os
import socket
#from skimage import io
import json
import base64
from io import StringIO, BytesIO
import zlib


def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default="localhost", help='sergver address')
    parser.add_argument('--port', '-p', default=4000, type=int, help='port')
    parser.add_argument('--img', nargs=1, type=str, help='image files')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='debug mode if this flag is set (default: False)')
    return parser.parse_args()


def main():
    args = arg()
    senddict = {}
    imgfile = args.img[0]

    senddict['frame'] = str(5)
    with open(imgfile, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
        print(type(encoded_string))

    senddict['img'] = encoded_string.decode('utf-8')

    if args.verbose:
        print(senddict)
    img_json = json.dumps(senddict)


    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((args.host, args.port))
    img_json = img_json.encode('utf-8')

    status = client.sendall(img_json)
    print('byte: {},  status: {}'.format(len(img_json), status))

    response = client.recv(4096).decode('utf-8')
    print(response)


if __name__ == '__main__':
    main()
